import time
from collections import deque

class TimeManager:
    def __init__(self, n_mean = 20):
        self.n_mean = n_mean
        self.reset()
    
    def reset(self):
        self.timedict = dict()
    
    def start(self, keyword):
        if keyword not in self.timedict:
            self.timedict[keyword] = {
                'start_timestamp': time.time(),
                'deque': deque(maxlen=self.n_mean),
                'mean': -1,
                'last_time': -1,
            }
        else:
            self.timedict[keyword]['start_timestamp'] = time.time()
    
    def end(self, keyword):
        if keyword in self.timedict:
            time_current = time.time() - self.timedict[keyword]['start_timestamp']
            self.timedict[keyword]['last_time'] = time_current
            self.timedict[keyword]['deque'].append(time_current)
            self.timedict[keyword]['start_timestamp'] = -1
            self.timedict[keyword]['mean'] = sum(self.timedict[keyword]['deque']) / len(self.timedict[keyword]['deque'])
            
            return self.timedict[keyword]['last_time'], self.timedict[keyword]['mean']
        
    def get_statistics(self):
        return {k: self.timedict[k]['mean'] for k in self.timedict}