import cv2 
import numpy as np 
import datetime

from torch.utils.tensorboard import SummaryWriter

class LogManager:
    def __init__(self):
        self.initialized = False
        self.history = dict()
        
    def append(self, result):
        if not self.initialized:
            for key in result.keys():
                self.history[key] = []
            self.initialized = True
        for key in result.keys():
            self.history[key].append(result[key])
    
    def get_statistics(self, mode='mean'):
        ret = dict()
        if mode == 'mean':
            for key in self.history.keys():
                item = self.history[key]
                ret[key] = 0 if len(item) == 0 else round(sum(item)/len(item), 4)
                self.history[key].clear()
        return ret
    
class BoardManager:
    def __init__(self, config_agent, config_env):
        self.date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
        self.save_path = './saved_models/{}/{}_{}/'.format(config_env['name'], self.date_time, config_agent['name'])
        self.writer = SummaryWriter('{}'.format(self.save_path))
        
    def write_scalar(self, mean_score, step):
        self.writer.add_scalar('Mean_Score', mean_score, step)    