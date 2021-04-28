from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size=None):
        self.buffer = list() if buffer_size is None else deque(maxlen=buffer_size)
        self.first_store = True
    
    def store(self, state, action, reward, next_state, done):
        if self.first_store:
            print("########################################")
            print("You should check dimension of transition")
            print("state:", state.shape)
            print("action:", action.shape)
            print("reward:", reward.shape)
            print("next_state:", next_state.shape)
            print("done:", done.shape)
            print("########################################")
            self.first_store = False
            
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
    
    def rollout(self):
        state       = np.stack([b[0] for b in self.buffer], axis=0)
        action      = np.stack([b[1] for b in self.buffer], axis=0)
        reward      = np.stack([b[2] for b in self.buffer], axis=0)
        next_state  = np.stack([b[3] for b in self.buffer], axis=0)
        done        = np.stack([b[4] for b in self.buffer], axis=0)
        
        self.clear()
        
        return (state, action, reward, next_state, done)
   
    def clear(self):
        self.buffer.clear()
    
    @property
    def size(self):
        return len(self.buffer)

# Reference: https://github.com/LeejwUniverse/following_deepmid/tree/master/jungwoolee_pytorch/100%20Algorithm_For_RL/01%20sum_tree
class PERBuffer(ReplayBuffer):
    def __init__(self, batch_size, buffer_size):
        self.buffer = [0 for i in range(buffer_size)] # define replay buffer
        self.sum_tree = [0 for i in range((buffer_size * 2) - 1)] # define sum tree
        
        self.tree_index = buffer_size - 1 # define sum_tree leaf node index.
        self.buffer_index = 0 # define replay buffer index.
        self.buffer_counter = 0
        
        self.batch_size = batch_size
        self.buffer_size = buffer_size 
        self.first_store = True
        
        self.max_priority = 1.0
        self.min_priority = self.max_priority
        
    def store(self, state, action, reward, next_state, done):
        if self.first_store:
            print("########################################")
            print("You should check dimension of transition")
            print("state:", state.shape)
            print("action:", action.shape)
            print("reward:", reward.shape)
            print("next_state:", next_state.shape)
            print("done:", done.shape)
            print("########################################")
            self.first_store = False
        
        for s, a, r, ns, d in zip(state, action, reward, next_state, done):
            self.buffer[self.buffer_index] = (s, a, r, ns, d)
        
            self.add_tree_data()

            self.buffer_index += 1
            self.buffer_counter += 1

            self.buffer_counter = min(self.buffer_counter, self.buffer_size)
            self.buffer_index = self.buffer_index % self.buffer_size
        
    def add_tree_data(self):
        if self.tree_index == (self.buffer_size * 2) - 1: # if sum tree index achive last index.
            self.tree_index = self.buffer_size - 1 # change frist leaf node index.
        
        self.update_priority(self.max_priority, self.tree_index)
        self.tree_index += 1 # count current sum_tree index

    def update_tree(self, index):
        # index is a starting leaf node point.
        while True: 
            index = (index - 1)//2 # parent node index.
            left = (index * 2) + 1 # left child node inex.
            right = (index * 2) + 2 # right child node index
            
            self.sum_tree[index] = self.sum_tree[left] + self.sum_tree[right] # sum both child node.
            if index == 0: ## if index is a root node.
                break

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
            
            if index >= self.buffer_size -1:
                break

        priority = self.sum_tree[index]
        tree_idx = index 
        buffer_idx = index - (self.buffer_size - 1)
        
        return priority, tree_idx, buffer_idx
    
    def sample(self, beta):
        batch = []
        idx_batch = []
        w_batch = np.zeros(self.batch_size)
        
        sum_p = self.sum_tree[0] 
        min_p = self.min_priority/sum_p
        max_w = pow(self.buffer_size * min_p, -beta)
        
        seg_size = sum_p/self.batch_size

        for i in range(self.batch_size):
            seg1 = seg_size * i
            seg2 = seg_size * (i + 1)

            sampled_val = np.random.uniform(seg1, seg2)
            priority, tree_idx, buffer_idx = self.search_tree(sampled_val)
            
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
    
    def update_priority(self, priority, index):
        self.sum_tree[index] = priority
        self.update_tree(index)
        
        self.min_priority = min(self.min_priority, priority)
        self.max_priority = max(self.max_priority, priority)
        
    @property
    def size(self):
        return len(self.buffer)
