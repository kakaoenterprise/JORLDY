from collections import deque
import random
import torch

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
    
    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        
        state       = torch.tensor([b[0] for b in batch]).float().view(batch_size, -1)
        action      = torch.tensor([b[1] for b in batch]).float().view(batch_size, -1)
        reward      = torch.tensor([b[2] for b in batch]).float().view(batch_size, -1)
        next_state  = torch.tensor([b[3] for b in batch]).float().view(batch_size, -1)
        done        = torch.tensor([b[4] for b in batch]).float().view(batch_size, -1)

        return (state, action, reward, next_state, done)
    
    @property
    def length(self):
        return len(self.buffer)
