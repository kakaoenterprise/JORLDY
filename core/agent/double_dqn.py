import torch
import torch.nn.functional as F
import random
import os

from .dqn import DQNAgent
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DoubleDQNAgent(DQNAgent):
    def __init__(self, **kwargs):
        super(DoubleDQNAgent, self).__init__(**kwargs)

    def learn(self):
        pass
        '''
        if self.memory.length < max(self.batch_size, self.start_train_step):
            return None
        
        transitions = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device), transitions)
        
        eye = torch.eye(self.action_size).to(device)
        one_hot_action = eye[action.view(-1).long()]
        q = (self.network(state) * one_hot_action).sum(1, keepdims=True)
        
        with torch.no_grad():
            max_Q = torch.max(q).item()
            next_q = self.target_network(next_state)
            target_q = reward + next_q.max(1, keepdims=True).values * (self.gamma*(1 - done))
        
        loss = F.smooth_l1_loss(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.num_learn % self.target_update_term == 0:
            self.update_target()
        self.num_learn += 1
        
        result = {
            "loss" : loss.item(),
            "epsilon" : self.epsilon,
            "max_Q": max_Q,
        }
        return result
        '''