from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        self.first_store = True
    
    def check_dim(self, transition):
        print("########################################")
        print("You should check dimension of transition")
        for key, val in transition.items():
            print(f"{key}: {val.shape}")
        print("########################################")
        self.first_store = False
            
    def store(self, transitions):
        if self.first_store:
            self.check_dim(transitions[0])
        self.buffer += transitions

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        transitions = {}
        for key in batch[0].keys():
            transitions[key] = np.stack([b[key][0] for b in batch], axis=0)
            
        return transitions
    
    def clear(self):
        self.buffer.clear()
    
    @property
    def size(self):
        return len(self.buffer)