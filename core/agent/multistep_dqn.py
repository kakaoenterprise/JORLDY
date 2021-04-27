import torch
import torch.nn.functional as F
import random
import os

from .dqn import DQNAgent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultistepDQNAgent(DQNAgent):
    def __init__(self, n_step = 5, **kwargs):
        super(MultistepDQNAgent, self).__init__(**kwargs)
        self.n_step = n_step
    
    def learn(self):
        if self.memory.size < max(self.batch_size, self.start_train_step):
            return None
        
#         shapes of 1-step implementations: (batch_size, dimension_data)
#         shapes of multistep implementations: (batch_size, steps, dimension_data)

        self.time_manager.start('time_learn')
        self.time_manager.start('time_sample')
        transitions = self.memory.sample_nstep(self.batch_size, self.n_step)
        self.time_manager.end('time_sample')
        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device), transitions)
        
        eye = torch.eye(self.action_size).to(device)
        one_hot_action = eye[action[:, 0].view(-1).long()]
#         q = (self.network(state[:, 0]) * one_hot_action).sum(1, keepdims=True)
        q = (self.network(state) * one_hot_action).sum(1, keepdims=True)
        with torch.no_grad():
            max_Q = torch.max(q).item()
#             next_q = self.target_network(next_state[:, -1])
            next_q = self.target_network(next_state)
            target_q = next_q.max(1, keepdims=True).values

            for i in reversed(range(self.n_step)):
                target_q = reward[:, i] + (1 - done[:, i]) * self.gamma * target_q
            
        loss = F.smooth_l1_loss(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.num_learn += 1
        
        self.time_manager.end('time_learn')

        result = {
            "loss" : loss.item(),
            "epsilon" : self.epsilon,
            "max_Q": max_Q,
        }
        stats_time = self.time_manager.get_statistics()
        for k in stats_time.keys(): result[k] = stats_time[k]
        return result