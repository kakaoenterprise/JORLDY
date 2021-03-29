from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
    
    def store(self, transitions):
        if len(self.buffer) == 0:
            print("########################################")
            print("You should check dimension of transition")
            print("state:",      transitions[0][0].shape)
            print("action:",     transitions[0][1].shape)
            print("reward:",     transitions[0][2].shape)
            print("next_state:", transitions[0][3].shape)
            print("done:",       transitions[0][4].shape)
            print("########################################")
            
        self.buffer += transitions

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        
        state       = np.concatenate([b[0] for b in batch], axis=0)
        action      = np.concatenate([b[1] for b in batch], axis=0)
        reward      = np.concatenate([b[2] for b in batch], axis=0)
        next_state  = np.concatenate([b[3] for b in batch], axis=0)
        done        = np.concatenate([b[4] for b in batch], axis=0)
        
        return (state, action, reward, next_state, done)
    
    def clear(self):
        self.buffer.clear()
    
    @property
    def size(self):
        return len(self.buffer)
    
# class ReplayBuffer:
#     def __init__(self, state_dim, action_dim, buffer_size):
#         self.buffer_size = buffer_size
#         self.ptr = 0
#         self.size = 0

#         self.state = np.zeros((buffer_size, *state_dim))
#         self.action = np.zeros((buffer_size, *action_dim))
#         self.next_state = np.zeros((buffer_size, *state_dim))
#         self.reward = np.zeros((buffer_size, 1))
#         self.done = np.zeros((buffer_size, 1))
    
#     def store(self, state, action, reward, next_state, done):
#         _size = reward.shape[0]
        
#         if _size > self.buffer_size:
#             print("Buffer size is too small!")
#             exit()
        
#         over_size = max(0, self.ptr + _size - self.buffer_size)
#         if over_size > 0:
#             remain_size = self.buffer_size - self.ptr
#             self.state[self.ptr:self.buffer_size] = state[:remain_size]
#             self.action[self.ptr:self.buffer_size] = action[:remain_size]
#             self.reward[self.ptr:self.buffer_size] = reward[:remain_size]
#             self.next_state[self.ptr:self.buffer_size] = next_state[:remain_size]
#             self.done[self.ptr:self.buffer_size] = done[:remain_size]
            
#             self.state[:_size - remain_size] = state[remain_size:]
#             self.action[:_size - remain_size] = action[remain_size:]
#             self.reward[:_size - remain_size] = reward[remain_size:]
#             self.next_state[:_size - remain_size] = next_state[remain_size:]
#             self.done[:_size - remain_size] = done[remain_size:]
            
#             self.ptr = _size - remain_size
#             self.size = self.buffer_size
#         else:
#             self.state[self.ptr:self.ptr+_size] = state
#             self.action[self.ptr:self.ptr+_size] = action
#             self.reward[self.ptr:self.ptr+_size] = reward
#             self.next_state[self.ptr:self.ptr+_size] = next_state
#             self.done[self.ptr:self.ptr+_size] = done

#             self.ptr = (self.ptr + _size) % self.buffer_size
#             self.size = min(self.size + _size, self.buffer_size)

#     def sample(self, batch_size):
#         batch_idx = np.random.choice(self.size, size=batch_size, replace=False)
        
#         state       = self.state[batch_idx]
#         action      = self.action[batch_idx]
#         reward      = self.reward[batch_idx]
#         next_state  = self.next_state[batch_idx]
#         done        = self.done[batch_idx]
        
#         return (state, action, reward, next_state, done)
    


