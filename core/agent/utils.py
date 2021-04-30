from collections import deque
import random
import numpy as np
import itertools

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        self.first_store = True
    
    def check_dim(self, state, action, reward, next_state, done):
        print("########################################")
        print("You should check dimension of transition")
        print("state:", state.shape)
        print("action:", action.shape)
        print("reward:", reward.shape)
        print("next_state:", next_state.shape)
        print("done:", done.shape)
        print("########################################")
        self.first_store = False
            
    def store(self, state, action, reward, next_state, done):
        if self.first_store:
            self.check_dim(state, action, reward, next_state, done)
        
        for s, a, r, ns, d in zip(state, action, reward, next_state, done):
            self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        
        state       = np.stack([b[0] for b in batch], axis=0)
        action      = np.stack([b[1] for b in batch], axis=0)
        reward      = np.stack([b[2] for b in batch], axis=0)
        next_state  = np.stack([b[3] for b in batch], axis=0)
        done        = np.stack([b[4] for b in batch], axis=0)
        
        return (state, action, reward, next_state, done)
    
    def clear(self):
        self.buffer.clear()
    
    @property
    def size(self):
        return len(self.buffer)

class MultistepBuffer(ReplayBuffer):
    def __init__(self, buffer_size, n_step):
        super(MultistepBuffer, self).__init__(buffer_size)
        self.n_step = n_step
        self.buffer_nstep = deque(maxlen=n_step)
        
    def prepare_nstep(self, batch):
        state = batch[0][0]
        next_state = batch[-1][3]

        action = np.stack([b[1] for b in batch], axis = 0)
        reward = np.stack([b[2] for b in batch], axis = 0)
        done = np.stack([b[4] for b in batch], axis = 0)

        return (state, action, reward, next_state, done)
        
    def store(self, state, action, reward, next_state, done):
        if self.first_store:
            self.check_dim(state, action, reward, next_state, done)
        
        for s, a, r, ns, d in zip(state, action, reward, next_state, done):
            self.buffer_nstep.append((s, a, r, ns, d))
            if len(self.buffer_nstep) == self.buffer_nstep.maxlen:
                self.buffer.append(self.prepare_nstep(self.buffer_nstep))
                
# Reference: https://github.com/LeejwUniverse/following_deepmid/tree/master/jungwoolee_pytorch/100%20Algorithm_For_RL/01%20sum_tree
class PERBuffer(ReplayBuffer):
    def __init__(self, buffer_size):
        self.buffer = [0 for i in range(buffer_size)] # define replay buffer
        self.sum_tree = [0 for i in range((buffer_size * 2) - 1)] # define sum tree
        
        self.tree_index = buffer_size - 1 # define sum_tree leaf node index.
        self.buffer_index = 0 # define replay buffer index.
        self.buffer_counter = 0
        
        self.buffer_size = buffer_size 
        self.first_store = True
        self.full_charge = False
        
        self.max_priority = 1.0
        self.min_priority = self.max_priority
        
    def store(self, state, action, reward, next_state, done):
        if self.first_store:
            self.check_dim(state, action, reward, next_state, done)
        
        for s, a, r, ns, d in zip(state, action, reward, next_state, done):
            self.buffer[self.buffer_index] = (s, a, r, ns, d)
            self.add_tree_data()

            self.buffer_counter = min(self.buffer_counter + 1, self.buffer_size)
            self.buffer_index = (self.buffer_index + 1) % self.buffer_size
                
    def add_tree_data(self):
        self.update_priority(self.max_priority, self.tree_index)
        self.tree_index += 1 # count current sum_tree index

        if self.tree_index == (self.buffer_size * 2) - 1: # if sum tree index achive last index.
            self.tree_index = self.buffer_size - 1 # change frist leaf node index.
            self.full_charge = True

    def update_tree(self, index, delta_priority):
        # index is a starting leaf node point.
        while index != 0: 
            index = (index - 1)//2 # parent node index.
            self.sum_tree[index] += delta_priority
    
    def search_tree(self, num):
        index = 0 # always start from root index.
        while True:
            left = (index * 2) + 1
            right = (index * 2) + 2
            
            if num <= self.sum_tree[left]: # if child left node is over current value. 
                index = left                # go to the left direction.
            else:
                num -= self.sum_tree[left] # if child left node is under current value.
                index = right               # go to the right direction.
            
            if index >= self.buffer_size - 1:
                break

        priority = self.sum_tree[index]
        tree_idx = index 
        buffer_idx = index - (self.buffer_size - 1)
        
        return priority, tree_idx, buffer_idx
    
    def sample(self, beta, batch_size):
        batch = []
        idx_batch = []
        w_batch = np.zeros(batch_size)
        
        sum_p = self.sum_tree[0] 
        min_p = self.min_priority/sum_p
        max_w = pow(self.buffer_size * min_p, -beta)
        
        seg_size = sum_p/batch_size
        
        priority_list = []
        
        for i in range(batch_size):
            seg1 = seg_size * i
            seg2 = seg_size * (i + 1)

            sampled_val = np.random.uniform(seg1, seg2)
            priority, tree_idx, buffer_idx = self.search_tree(sampled_val)
            
            priority_list.append(priority)
            
            batch.append(self.buffer[buffer_idx])
            idx_batch.append(tree_idx)
            
            p_i = priority/sum_p
            w_i = pow((self.buffer_size * p_i), -beta)
            w_batch[i] = w_i/max_w
        
        state       = np.stack([b[0] for b in batch], axis=0)
        action      = np.stack([b[1] for b in batch], axis=0)
        reward      = np.stack([b[2] for b in batch], axis=0)
        next_state  = np.stack([b[3] for b in batch], axis=0)
        done        = np.stack([b[4] for b in batch], axis=0)
        
        return (state, action, reward, next_state, done), w_batch, idx_batch
    
    def update_priority(self, new_priority, index):
        ex_priority = self.sum_tree[index]
        delta_priority = new_priority - ex_priority
        self.sum_tree[index] = new_priority
        self.update_tree(index, delta_priority)
        
        if self.full_charge:
            if self.min_priority != ex_priority:
                self.min_priority = min(self.min_priority, new_priority)
            else:
                min(self.sum_tree[self.buffer_size - 1:])
            
            if self.max_priority != ex_priority:
                self.max_priority = max(self.max_priority, new_priority)
            else:
                max(self.sum_tree[self.buffer_size - 1:])

class Rollout(ReplayBuffer):
    def __init__(self, **kwargs):
        self.buffer = list()
        self.first_store = False
    
    def rollout(self):
        state       = np.stack([b[0] for b in self.buffer], axis=0)
        action      = np.stack([b[1] for b in self.buffer], axis=0)
        reward      = np.stack([b[2] for b in self.buffer], axis=0)
        next_state  = np.stack([b[3] for b in self.buffer], axis=0)
        done        = np.stack([b[4] for b in self.buffer], axis=0)
        
        self.clear()
        return (state, action, reward, next_state, done)