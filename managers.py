import numpy as np 
import datetime
import copy
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter

class MetricManager:
    def __init__(self):
        self.metrics = defaultdict(list)
        
    def append(self, result):
        for key, value in result.items():
            self.metrics[key].append(value)
    
    def get_statistics(self, mode='mean'):
        ret = dict()
        if mode == 'mean':
            for key, value in self.metrics.items():
                ret[key] = 0 if len(value) == 0 else round(sum(value)/len(value), 4)
                self.metrics[key].clear()
        return ret
    
class LogManager:
    def __init__(self, env, id):
        self.id=id
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.path = f"./logs/{env}/{id}/{now}/"
        self.writer = SummaryWriter(self.path)
        
    def write_scalar(self, scalar_dict, step):
        for key, value in scalar_dict.items():
            self.writer.add_scalar(f"{self.id}/"+key, value, step)
            self.writer.add_scalar("all/"+key, value, step)
            
            
class TestManager:
    def __init__(self):
        pass
    
    def test(self, agent, env, iteration=10):
        if not iteration > 0:
            print("Error!!! test iteration is not > 0")
            return 0
        
        scores = []
        for i in range(iteration):
            done = False
            state = env.reset()
            while not done:
                action = agent.act(state, training=False)
                state, reward, done = env.step(action)
            scores.append(env.score)
            
        return np.mean(scores)
        
                
        
        
        