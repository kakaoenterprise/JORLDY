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
        
#         state       = torch.cat([torch.tensor([b[0]]).float() for b in batch])
#         action      = torch.cat([torch.tensor([b[1]]).float() for b in batch])
#         reward      = torch.cat([torch.tensor([b[2]]).float() for b in batch])
#         next_state  = torch.cat([torch.tensor([b[3]]).float() for b in batch])
#         done        = torch.cat([torch.tensor([b[4]]).float() for b in batch])
        
        state_batch = torch.cat([torch.tensor([batch[i][0]]) for i in range(batch_size)]).float()
        action_batch = torch.cat([torch.tensor([batch[i][1]]) for i in range(batch_size)]).float()
        reward_batch = torch.cat([torch.tensor([batch[i][2]]) for i in range(batch_size)]).float()
        next_state_batch = torch.cat([torch.tensor([batch[i][3]]) for i in range(batch_size)]).float()
        done_batch = torch.cat([torch.tensor([batch[i][4]]) for i in range(batch_size)]).float()

#         transitions = []
#         for item in [state, action, reward, next_state, done]:
#             if len(list(item.shape)) == 1:
#                 item = torch.unsqueeze(item, 1) 
#             transitions.append(item)
        transitions = [state_batch, action_batch, reward_batch, next_state_batch, done_batch]
    
        return transitions
    
    @property
    def length(self):
        return len(self.buffer)

