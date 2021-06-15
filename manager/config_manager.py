class ConfigManager:
    def __init__(self, config_path):
        module = __import__(config_path, fromlist=[None])
        self.config = CustomDict()
        self.config.agent = CustomDict(module.agent)
        self.config.env = CustomDict(module.env)
        self.config.train = CustomDict(module.train)
    
class CustomDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __getitem__ = __getattr__
    
    def __init__(self, init_dict={}):
        self.update(init_dict)
    