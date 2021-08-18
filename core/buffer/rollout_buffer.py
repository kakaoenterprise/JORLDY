import numpy as np

from .replay_buffer import ReplayBuffer

class RolloutBuffer(ReplayBuffer):
    def __init__(self, **kwargs):
        self.buffer = list()
        self.first_store = True
    
    def rollout(self):
        transitions = {}
        for key in self.buffer[0].keys():
            transitions[key] = np.stack([b[key][0] for b in self.buffer], axis=0)
            
        self.clear()
        return transitions