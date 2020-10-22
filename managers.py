import cv2 
import numpy as np 
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
    def __init__(self, save_path):
        self.writer = SummaryWriter('{}'.format(save_path))
        
    def write_scalar(self, mean_score, step):
        self.writer.add_scalar('Mean_Score', mean_score, step)    