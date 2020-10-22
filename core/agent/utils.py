from collections import deque
import random
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
    
    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        
        state       = np.stack([b[0] for b in batch], axis=0)
        action      = np.stack([b[1] for b in batch], axis=0)
        reward      = np.stack([b[2] for b in batch], axis=0)
        next_state  = np.stack([b[3] for b in batch], axis=0)
        done        = np.stack([b[4] for b in batch], axis=0)
        
        transition = []
        for item in [state, action, reward, next_state, done]:
            item = torch.tensor(item).float()
            if len(list(item.shape)) == 1:
                item = torch.unsqueeze(item, 1)
            transition.append(item)
                
        return transition
    
    @property
    def length(self):
        return len(self.buffer)

