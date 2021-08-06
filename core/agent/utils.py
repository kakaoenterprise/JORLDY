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

class MultistepBuffer(ReplayBuffer):
    def __init__(self, buffer_size, n_step):
        super(MultistepBuffer, self).__init__(buffer_size)
        self.n_step = n_step
        
    def prepare_nstep(self, batch):
        transition = {}
        transition['state'] = batch[0]['state']
        transition['next_state'] = batch[-1]['next_state']
        
        for key in batch[0].keys():
            if key not in ['state', 'next_state']:
                transition[key] = np.stack([b[key] for b in batch], axis=1)
        
        return transition
        
    def store(self, transitions, delta_t=1):
        if self.first_store:
            self.check_dim(transitions[0])
            self.nstep_buffers = [deque(maxlen=self.n_step) for _ in range(len(transitions)//delta_t)]
        
        # Issue: need to consider multiple actor
        for i, transition in enumerate(transitions):
            nstep_buffer = self.nstep_buffers[i//delta_t]
            nstep_buffer.append(transition)
            if len(nstep_buffer) == self.n_step:
                self.buffer.append(self.prepare_nstep(nstep_buffer))
                
# Reference: https://github.com/LeejwUniverse/following_deepmid/tree/master/jungwoolee_pytorch/100%20Algorithm_For_RL/01%20sum_tree
class PERBuffer(ReplayBuffer):
    def __init__(self, buffer_size, uniform_sample_prob=1e-3):
        self.buffer_size = buffer_size 
        self.tree_size = (buffer_size * 2) - 1
        self.first_leaf_index = buffer_size - 1 
        
        self.buffer = np.zeros(buffer_size, dtype=tuple) # define replay buffer
        self.sum_tree = np.zeros(self.tree_size) # define sum tree
        
        self.buffer_index = 0 # define replay buffer index.
        self.tree_index = self.first_leaf_index # define sum_tree leaf node index.
        self.buffer_counter = 0
        self.max_priority = 1.0
        self.uniform_sample_prob = uniform_sample_prob
        
        self.first_store = True
        
    def store(self, transitions):
        if self.first_store:
            self.check_dim(transitions[0])
        
        for transition in transitions:
            self.buffer[self.buffer_index] = transition
            self.add_tree_data()

            self.buffer_counter = min(self.buffer_counter + 1, self.buffer_size)
            self.buffer_index = (self.buffer_index + 1) % self.buffer_size
                
    def add_tree_data(self):
        self.update_priority(self.max_priority, self.tree_index)

        self.tree_index += 1 # count current sum_tree index
        if self.tree_index == self.tree_size: # if sum tree index achive last index.
            self.tree_index = self.first_leaf_index # change frist leaf node index.
    
    def update_priority(self, new_priority, index):
        ex_priority = self.sum_tree[index]
        delta_priority = new_priority - ex_priority
        self.sum_tree[index] = new_priority
        self.update_tree(index, delta_priority)

        self.max_priority = max(self.max_priority, new_priority)

    def update_tree(self, index, delta_priority):
        # index is a starting leaf node point.
        while index > 0: 
            index = (index - 1)//2 # parent node index.
            self.sum_tree[index] += delta_priority
            
    def search_tree(self, num):
        index = 0 # always start from root index.
        while index < self.first_leaf_index:
            left = (index * 2) + 1
            right = (index * 2) + 2
            
            if num <= self.sum_tree[left]: # if child left node is over current value. 
                index = left                # go to the left direction.
            else:
                num -= self.sum_tree[left] # if child left node is under current value.
                index = right               # go to the right direction.

        return index
    
    def sample(self, beta, batch_size):
        assert self.sum_tree[0] > 0.
        uniform_sampling = np.random.uniform(size=batch_size) < self.uniform_sample_prob
        uniform_size = np.sum(uniform_sampling)
        prioritized_size = batch_size - uniform_size
        
        uniform_indices = list(np.random.randint(self.buffer_counter, size=uniform_size) + self.first_leaf_index)
        
        targets = np.random.uniform(size=prioritized_size) * self.sum_tree[0]
        prioritized_indices = [self.search_tree(target) for target in targets]
        
        indices = np.asarray(uniform_indices + prioritized_indices)
        priorities = np.asarray([self.sum_tree[index] for index in indices])
        assert len(indices) == len(priorities) == batch_size
        
        uniform_probs = np.asarray(1. / self.buffer_counter)
        prioritized_probs = priorities / self.sum_tree[0]
        
        usp = self.uniform_sample_prob
        sample_probs = (1. - usp) * prioritized_probs + usp * uniform_probs
        weights = (uniform_probs / sample_probs) ** beta
        weights /= np.max(weights)
        batch = [self.buffer[idx] for idx in indices - self.first_leaf_index]
        
        transitions = {}
        for key in batch[0].keys():
            transitions[key] = np.stack([b[key][0] for b in batch], axis=0)
            
        sampled_p = np.mean(priorities) 
        mean_p = self.sum_tree[0]/self.buffer_counter
        return transitions, weights, indices, sampled_p, mean_p
    
    @property
    def size(self):
        return self.buffer_counter
    
class Rollout(ReplayBuffer):
    def __init__(self, **kwargs):
        self.buffer = list()
        self.first_store = True
    
    def rollout(self):
        transitions = {}
        for key in self.buffer[0].keys():
            transitions[key] = np.stack([b[key][0] for b in self.buffer], axis=0)
            
        self.clear()
        return transitions
    
# Reference: https://github.com/LeejwUniverse/following_deepmid/tree/master/jungwoolee_pytorch/100%20Algorithm_For_RL/01%20sum_tree
class PERMultistepBuffer(PERBuffer, MultistepBuffer):
    def __init__(self, buffer_size, n_step, uniform_sample_prob=1e-3):
        MultistepBuffer.__init__(self, buffer_size, n_step)
        PERBuffer.__init__(self, buffer_size, uniform_sample_prob)
        
    def store(self, transitions, delta_t=1):
        if self.first_store:
            self.check_dim(transitions[0])
            self.nstep_buffers = [deque(maxlen=self.n_step) for _ in range(len(transitions)//delta_t)]
        
        for i, transition in enumerate(transitions):
            # MultiStep
            nstep_buffer = self.nstep_buffers[i//delta_t]
            nstep_buffer.append(transition)
            if len(nstep_buffer) == self.n_step:
                self.buffer[self.buffer_index] = self.prepare_nstep(nstep_buffer)
        
                # PER
                self.add_tree_data()

                self.buffer_counter = min(self.buffer_counter + 1, self.buffer_size)
                self.buffer_index = (self.buffer_index + 1) % self.buffer_size
    