import torch
import torch.nn.functional as F

from .utils import MultistepBuffer
from .dqn import DQNAgent

class MultistepDQNAgent(DQNAgent):
    def __init__(self, n_step = 5, **kwargs):
        super(MultistepDQNAgent, self).__init__(**kwargs)
        self.n_step = n_step
        self.memory = MultistepBuffer(self.buffer_size, self.n_step)
    
    def learn(self):
#         shapes of 1-step implementations: (batch_size, dimension_data)
#         shapes of multistep implementations: (batch_size, steps, dimension_data)

        transitions = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(self.device), transitions)
        
        eye = torch.eye(self.action_size).to(self.device)
        one_hot_action = eye[action[:, 0].view(-1).long()]
        q = (self.network(state) * one_hot_action).sum(1, keepdims=True)
        with torch.no_grad():
            max_Q = torch.max(q).item()
            next_q = self.target_network(next_state)
            target_q = next_q.max(1, keepdims=True).values

            for i in reversed(range(self.n_step)):
                target_q = reward[:, i] + (1 - done[:, i]) * self.gamma * target_q
            
        loss = F.smooth_l1_loss(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.num_learn += 1

        result = {
            "loss" : loss.item(),
            "epsilon" : self.epsilon,
            "max_Q": max_Q,
        }
        
        return result