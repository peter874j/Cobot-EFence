import argparse
import time
from pathlib import Path
from datetime import datetime

import cv2,os
import torch
import torch.backends.cudnn as cudnn
from skimage.metrics import structural_similarity
from interval import Interval
from numpy import random
import configparser
from ssim import ssim
from abs import Abs
import matplotlib.path as mpltPath
import numpy as np
from os import walk

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import time
#from LinkPost import *


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    videopath=r'./video/'
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    camNo = 86
    camn = 0  
    source = []
    dataset = []
    for i in range(10):
        source.append('rtsp://192.168.137.{}/h265'.format(camNo + i))
    #source2 = 'rtsp://192.168.137.{}/h265'.format(camNo + 1)
    webcam = True
    #source = root + f
    # Set Dataloader
    print()
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        for i in source:
            dataset.append(LoadStreams(i, img_size=imgsz))
        #dataset2 = LoadStreams(source2, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    while True:
        start = datetime.now()
        for path, img, im0s, vid_cap in dataset[camn]:
            orgim=im0s
            t1 = time_synchronized()        
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference

            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                violation_frame=[]
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset[camn].count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset[camn], 'frame', 0)
                SrvCfg = configparser.ConfigParser()
                SrvCfg.read('./L3Bmask.cfg')
                CAM = 'CAM_1_'
                orgpath = SrvCfg.get(CAM, 'orgpath')  
                goodshight = eval(SrvCfg.get(CAM,'goodshight'))
                zebraglue = eval(SrvCfg.get(CAM,'zebraglue'))
                limit = eval(SrvCfg.get(CAM,'limit'))
                margin = opt.margin
                goodshight_margin = [(goodshight[0][0], goodshight[0][1]), (goodshight[1][0], goodshight[1][1]), (goodshight[2][0], goodshight[2][1] - margin), (goodshight[3][0], goodshight[3][1] - margin)]
                zebraglue_margin = [(zebraglue[0][0], zebraglue[0][1] - margin), (zebraglue[1][0], zebraglue[1][1] - margin), (zebraglue[2][0], zebraglue[2][1] + margin), (zebraglue[3][0], zebraglue[3][1] + margin)]
                limit_margin = [(limit[0][0] + margin, limit[0][1] - margin), (limit[1][0] - margin, limit[1][1] - margin), (limit[2][0] - margin, limit[2][1] + margin), (limit[3][0] + margin, limit[3][1] + margin)]          
                #abnormal = ssim(orgpath, im0s, source)
                abnormal = Abs(orgpath, im0, source[camn], goodshight, zebraglue, limit)
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset[camn].mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f'{n} {names[int(c)]}s, '  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            if names[int(cls)] == 'Box':
                                color=(255,0,0)
                            elif names[int(cls)] == 'YBox':
                                color=(0,0,255)

                            violation=plot_one_box(xyxy, im0, abnormal, goodshight_margin, zebraglue_margin, limit_margin, names[int(cls)], color, line_thickness=2)
                            violation_frame.append(violation)
                now_localtime = time.strftime("%H:%M:%S", time.localtime())
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(im0, now_localtime, (50,50), font, 1.2, (0,0,0), 2)
                if  'Alarm' in violation_frame:
                    print('Alarm')
                    #Linkpost(im0s)
                    #time.sleep(100)

                else:
                    print('OK')
                # Print time (inference + NMS)
                # cv2.imshow('test',im0s)
                # cv2.waitKey(1)
                # time.sleep(2)
                #cv2.destroyAllWindows()
                t2 = time_synchronized()
                print(f'{s}Done. ({t2 - t1:.3f}s)')
                cv2.putText(im0, 'fps:' + str(round(1/(t2 - t1), 2)), (1000,50), font, 1.2, (0,0,0), 2)
                print(datetime.now())
                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)

                # Save results (image with detections)
                # if save_img:
                #     if dataset.mode == 'image':
                #         cv2.imwrite(save_path, im0)
                #     else:  # 'video'
                #         if vid_path != save_path:  # new video
                #             vid_path = save_path
                #             if isinstance(vid_writer, cv2.VideoWriter):
                #                 vid_writer.release()  # release previous video writer

                #             fourcc = 'mp4v'  # output video codec
                #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
                #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                #         vid_writer.write(im0)
            end = datetime.now()
            if (end - start).seconds     > 30:
                if camn == 9:
                    camn = 0
                else:
                    camn+=1
                cv2.destroyAllWindows()
                break
        # if save_txt or save_img:
        #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #     print(f"Results saved to {save_dir}{s}")


        print(f'All Done. ({time.time() - t0:.3f}s)')
        #os.remove(source)
        #             donevideo.append(source)
        # for i in donevideo:
        #     os.remove(i)
        # print("File removed successfully")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='L3B1028.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/test_data', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.05, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--margin', type=int, default=0, help='threshold for margin')    
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()