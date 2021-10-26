import numpy as np

from .base import BaseBuffer

class RolloutBuffer(BaseBuffer):
    def __init__(self):
        super(RolloutBuffer, self).__init__()
        self.buffer = list()
  
    def store(self, transitions):
        if self.first_store:
            self.check_dim(transitions[0])
        self.buffer += transitions 
        
    def sample(self):
        transitions = {}
        for key in self.buffer[0].keys():
            transitions[key] = np.stack([b[key][0] for b in self.buffer], axis=0)
            
        self.buffer.clear()
        return transitions
    
    @property
    def size(self):
        return len(self.buffer)
