import torch
import torch.nn.functional as F
import random
import os
from collections import deque

from .dqn import DQNAgent
from .utils import PERBuffer

class PERAgent(DQNAgent):
    def __init__(self, alpha, beta, learn_period=16, uniform_sample_prob=1e-3, **kwargs):
        super(PERAgent, self).__init__(**kwargs)
        self.memory = PERBuffer(self.buffer_size, uniform_sample_prob)
        self.alpha = alpha
        self.beta = beta 
        self.beta_add = 1/self.explore_step
        self.learn_period = learn_period
                
    def learn(self):        
        if self.memory.buffer_counter < max(self.batch_size, self.start_train_step):
            return None
                
        transitions, weights, indices, sampled_p, mean_p = self.memory.sample(self.beta, self.batch_size)
        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(self.device), transitions)
        
        eye = torch.eye(self.action_size).to(self.device)
        one_hot_action = eye[action.view(-1).long()]
        q = (self.network(state) * one_hot_action).sum(1, keepdims=True)
        
        with torch.no_grad():
            max_Q = torch.max(q).item()
            next_q = self.network(next_state)
            max_a = torch.argmax(next_q, axis=1)
            max_eye = torch.eye(self.action_size).to(self.device)
            max_one_hot_action = eye[max_a.view(-1).long()]
            
            next_target_q = self.target_network(next_state)
            target_q = reward + (next_target_q * max_one_hot_action).sum(1, keepdims=True) * (self.gamma*(1 - done))
                
        # Update sum tree
        td_error = abs(target_q - q)
        p_j = torch.pow(td_error, self.alpha)
        for i, p in zip(indices, p_j):
            self.memory.update_priority(p.item(), i)
                
        # Annealing beta
        self.beta = min(1.0, self.beta + self.beta_add)
        
        weights = torch.unsqueeze(torch.FloatTensor(weights).to(self.device), -1)
                
        loss = (weights * (td_error**2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.num_learn += 1

        result = {
            "loss" : loss.item(),
            "epsilon" : self.epsilon,
            "max_Q": max_Q,
            "sampled_p": sampled_p,
            "mean_p": mean_p,
        }
        return result

    def process(self, state, action, reward, next_state, done):
        result = None
        # Process per step
        self.memory.store(state, action, reward, next_state, done)
        
        if self.memory.size > 0 and self.memory.size % self.learn_period == 0:
            result = self.learn()

        # Process per step if train start
        if self.num_learn > 0:
            self.epsilon_decay()

            if self.num_learn % self.target_update_term == 0:
                self.update_target()
        
        # Process per episode
        if done.all():
            pass

        return result