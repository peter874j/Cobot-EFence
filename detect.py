import argparse
import time
import os
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from scipy.spatial import distance as dist

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box, plot_one_circle, plot_area, select_state_color
from utils.torch_utils import select_device, load_classifier, time_synchronized
from mark_points_on_image import MarkPoints
from config.config import Configuration as config
### ROS2 Package
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header 
from std_msgs.msg import Int32

class EfenceNode(Node):

    def __init__(self) :

        super().__init__('eFence')
        self.publisher = self.create_publisher(Int32, 'dangeZone', 10)

    def publish_zone(self, zone):

        msg = Int32()
        msg.data = zone   # 0->0%, 1->50%, 2->100%
        self.publisher.publish(msg)
        print("successfully published the zone value..." + str(msg.data))

    def destroy_node(self):
        self.destroy_node()
        rclpy.shutdown() 


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, remark = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.remark
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://'))

    ### Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    ### Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    initMarkFlag = False   # remark once
    msgDictionary = {}
    workingRadius = 0
    ### ROS Node Initial
    rclpy.init()
    ROS2Node = EfenceNode()
    preState = None
    curState = None
    
    ### Read cfg file name
    cfgFileName = r'./config.yml'
    caliArray = np.load(r'./R3V6F_720p_matrix.npz')
    ### print file creation time
    createTimePoint = int(os.stat(cfgFileName).st_ctime)
    # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(os.stat(cfgFileName).st_ctime)))

    ### Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    ### Second-stage classifier
    '''
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    '''
    ### Set Dataloader
    vid_writer = None
    if webcam:
        view_img = True
        save_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, caliArray, img_size=imgsz)
    else:
        save_img = True
        ### 無加上畸變校正
        dataset = LoadImages(source, img_size=imgsz)

    ### Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    ### Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        # print file modified time
        modifyTimePoint = int(os.stat(cfgFileName).st_mtime)
        # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(os.stat(cfgFileName).st_mtime)))
        modifyFlag = createTimePoint != modifyTimePoint
        createTimePoint = modifyTimePoint   # update create time point if modify 
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        ### Set Cobot Location & Working Radius
        ### manually set cobot location
        if remark and (not initMarkFlag):
            initMarkFlag = True
            cobotCoords = mark_points_on_camera_view_image(im0s)
            cv2.destroyAllWindows()
            workingRadius = 10
            print(cobotCoords)
            msgDictionary = {'cobot_coords':cobotCoords, 'working_radius':workingRadius}
            config.dump_config("./config.yml", msgDictionary)
        ### read cfg to set cobot location
        elif modifyFlag or (not initMarkFlag):
            initMarkFlag = True
            config.load_config("./config.yml")
            cobotCoords = config.cfg["cobot_coords"]
            distFactor = float(config.cfg["distFactor"])
            workingRadius = int(config.cfg["working_radius"])

        ### Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        ### Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        ### Apply Classifier
        '''
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        '''
        ### Process detections
        personState = 2
        stateDict = {'safe':0 ,'warn':0, 'stop':0}   # per frames state
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir)  if webcam else str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            distance = 0.

            if len(det):
                ### Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                ### Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string

                ### Write results
                for *xyxy, conf, cls in reversed(det):
                    bbox = []
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    
                    ### Add bbox to image
                    if save_img or view_img:  
                        label = f'{names[int(cls)]} {conf:.2f}'
                        bbox = [int(cls), int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])] # [class,x1,y1,x2,y2]
                        centerPoint = (int((xyxy[0] + xyxy[2])/2), int((xyxy[1] + xyxy[3])/2))
                        personRadius = int(max(abs(xyxy[0] - centerPoint[0]), abs(xyxy[1] - centerPoint[1])))
                        centerLinePoint = [int((centerPoint[0] + cobotCoords[0][0]) / 2), int((centerPoint[1] + cobotCoords[0][1]) / 2)]
                        ### Calculate Distance
                        distance, personState, stateLabel = calculate_distance(centerPoint, personRadius, cobotCoords[0], workingRadius, distFactor)
                        stateDict['safe'] = stateDict['safe'] + 1 if  personState==2 else stateDict['safe']
                        stateDict['warn'] = stateDict['warn'] + 1 if  personState==1 else stateDict['warn']
                        stateDict['stop'] = stateDict['stop'] + 1 if  personState==0 else stateDict['stop']
                        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        text = str(distance) + " m"
                        color = select_state_color(personState)
                        plot_one_circle(xyxy, im0, centerPoint, personRadius, personState, label = label, stateLabel = stateLabel, color = color, line_thickness = 3)
                        cv2.line(im0, centerPoint, cobotCoords[0], color = color, thickness = 3)   # drawn a line between cobot and person
                        cv2.putText(im0, text, centerLinePoint, cv2.FONT_HERSHEY_SIMPLEX, 1, color = color, thickness = 2, lineType = cv2.LINE_AA)
            ### plot start
            ### display corresponde safety level (safety, warning, danger)
            im0 = plot_area(im0, cobotCoords[0], workingRadius)   # mask add
            print("\n")
            print(f"distance: {distance}, personState: {personState}")        

            ### Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            ### save image or video results
            vid_writer = save_result(im0, save_path, vid_cap, vid_writer, dataset.mode, view_img, save_img)
        
        ### per frame state
        curState = respond_frame_state(stateDict)
        if curState != preState:
            preState = curState
            ROS2Node.publish_zone(curState)    # publish efence state to ROS2 topic 0:danger, 1:warning ,2:safety
            # rclpy.spin_once(ROS2Node, executor=None, timeout_sec = 0)
    ROS2Node.destroy_node()   # Destroy the node explicitly

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

def save_result(img, savePath, vid_cap, vid_writer, mode, view_img, save_img):
    """save image or video results

    Args:
        img ([type]): result image
        savePath ([type]): save path
        vid_cap ([type]): video capture
        vid_writer ([type]): video writer
        mode ([type]): image or video mode
        view_img ([type]): displaying flag
        save_img ([type]): saving flag
    """    
    # Stream results
    if view_img:
        cv2.imshow("display", img)
        # cv2.waitKey(1)  # 1 millisecond
    videoName = 'record.avi'
    # Save results (image with detections)
    if save_img:
        if mode == 'image':
            cv2.imwrite(savePath, img)
        else:  # 'video'
            if vid_writer == None:  # new video
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # output video codec
                # fps = min(15.0, vid_cap.get(cv2.CAP_PROP_FPS))   # NX limit 15 FPS
                # fps = vid_cap.get(cv2.CAP_PROP_FPS)   # NX limit 15 FPS
                # w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                # h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                # vid_writer = cv2.VideoWriter(savePath, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                vid_writer = cv2.VideoWriter(savePath, cv2.VideoWriter_fourcc(*fourcc), 12, (1280, 720))
                # vid_writer = cv2.VideoWriter(os.path.join(savePath, videoName), cv2.VideoWriter_fourcc(*fourcc), 12, (1280, 720))
            vid_writer.write(img)
    return vid_writer

### set cobot location
def mark_points_on_camera_view_image(img, numPoints=1):
    markPoints = MarkPoints(img, "Camera View")
    camera_view_points = markPoints.mark_points(numPoints)
    return camera_view_points

### respond frame state 
def respond_frame_state(stateDict):
    curState = None
    if stateDict['stop'] > 0:
        curState = 0
    elif stateDict['warn'] > 0:
        curState = 1
    elif stateDict['safe'] > 0:
        curState = 2
    else:
        curState = None
    return curState

### pixel convert truth distance
def convert_pixel_distance(pixelsNum, distFactor):
    truthDistance = round(pixelsNum * distFactor, 2)
    return truthDistance

### calculate distance of cobot between people
def calculate_distance(centerPoint, personRadius, cobotCoords, workingRadius, distFactor):
    """calculate distance of cobot between people

    Args:
        centerPoint (list): [x, y]
        personRadius (int):  Person Radius
        cobotCoords (list): [x, y]
        workingRadius (int): Working Radius

    Returns:
        [float]: distance
        [int]: stateCode
        [str]: stateLabel
    """    
    stateCode = 2
    stateLabel = "safety"
    totalpixelDistance = personRadius + workingRadius
    pixelDistance = round(dist.euclidean(centerPoint, cobotCoords), 2)    # euclidean distance
    distance = convert_pixel_distance(pixelDistance, distFactor)
    totalDistance = convert_pixel_distance(totalpixelDistance, distFactor)
    warningDistance = totalDistance + 0.3   # m
    ### danger state(stop)
    if distance <= totalDistance:
        stateCode = 0 
        stateLabel = "danger"
    ### safety state(50% speed)
    elif (distance > totalDistance) and (distance <= warningDistance):
        stateCode = 1
        stateLabel = "warning"
    ### safety state(100% speed)
    elif distance > warningDistance:
        stateCode = 2
        stateLabel = "safety"
    return distance, stateCode, stateLabel

def parse_all_argument():
    """Define All Parse Argument
    """    
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    # parser.add_argument('--weights', nargs='+', type=str, default='weights/Cobot_Sample_Weights.pt', help='model.pt path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default='weights/WY_yolov5s.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--source', type=str, default='20211129_Cobot_Demo.avi', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='record.avi', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--source', type=str, default='rtsp://192.168.137.97/h265', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.51, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--view-img', default=True, help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--remark', action='store_true', default=False, help='draw cobot location') 
    # parser.add_argument('--remark', action='store_true', default=True, help='draw cobot location') 
    
    return  parser.parse_args()

### ROS2 publisher 
def ROS2_publisher(args=None):

    
    rclpy.init(args=args)

    efence = Efence()

    while True:

        value = input("Choose Mode: ")
        value = value.lower()

        if( value == "s"):

            zone = 0
            efence.efence_publisher_func(zone)

        elif( value == "d"):

            zone = 1
            efence.efence_publisher_func(zone)

        elif( value == "f"):

            zone = 2
            efence.efence_publisher_func(zone)

        elif( value == "q"):

            break

        rclpy.spin_once(efence, executor=None, timeout_sec = 0)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    efence.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':

    opt = parse_all_argument()
    print(opt)

    ### Detect Start
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
