import torch
import torch.nn.functional as F
import random
import os
from collections import deque

from .dqn import DQNAgent
from .utils import PERBuffer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PERAgent(DQNAgent):
    def __init__(self, alpha, beta, eps, **kwargs):
        super(PERAgent, self).__init__(**kwargs)
        self.memory = PERBuffer(self.batch_size, self.buffer_size)
        self.alpha = alpha
        self.beta = beta 
        self.beta_add = 1/self.explore_step
        self.eps = eps
                
    def learn(self):        
        if self.memory.buffer_counter < max(self.batch_size, self.start_train_step):
            return None
                
        transitions, w_batch, idx_batch, sampled_p, mean_p = self.memory.sample(self.beta)
        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device), transitions)
                
        eye = torch.eye(self.action_size).to(device)
        one_hot_action = eye[action.view(-1).long()]
        q = (self.network(state) * one_hot_action).sum(1, keepdims=True)
        
        with torch.no_grad():
            max_Q = torch.max(q).item()
            next_q = self.network(next_state)
            max_a = torch.argmax(next_q, axis=1)
            max_eye = torch.eye(self.action_size).to(device)
            max_one_hot_action = eye[max_a.view(-1).long()]
            
            next_target_q = self.target_network(next_state)
            target_q = reward + (next_target_q * max_one_hot_action).sum(1, keepdims=True) * (self.gamma*(1 - done))
                
        # Update sum tree
        td_error = target_q - q
        p_j = torch.pow(torch.abs(td_error) + self.eps, self.alpha)
        
        for i, p in zip(idx_batch, p_j):
            self.memory.update_priority(p.item(), i)
                
        # Annealing beta
        if self.beta < 1:
            self.beta += self.beta_add
        else:
            self.beta = 1.0 
        
        w_batch = torch.FloatTensor(w_batch).to(device)
                
        loss = (w_batch * F.smooth_l1_loss(q, target_q, reduction="none")).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.num_learn += 1
        
        td_write = torch.abs(td_error).mean().item()
        
        result = {
            "loss" : loss.item(),
            "epsilon" : self.epsilon,
            "max_Q": max_Q,
            "sampled_p": sampled_p,
            "mean_p": mean_p
        }
        return result