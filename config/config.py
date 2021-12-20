import yaml


class Configuration:
    cfg = None

    @classmethod
    def load_config(cls, cfgPath):
        with open(cfgPath, "r") as ymlFile:
            cls.cfg = yaml.load(ymlFile, Loader = yaml.FullLoader)
    
    @classmethod
    def dump_config(cls, cfgPath, msgDictionary):
        with open(cfgPath, "w") as ymlFile:
            cls.cfg = yaml.dump(msgDictionary, ymlFile)