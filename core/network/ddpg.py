import torch
import torch.nn.functional as F
        
from .base import BaseNetwork

class DDPG_Critic(BaseNetwork):
    def __init__(self, D_in1, D_in2, head=None, D_hidden=512):
        D_in1, D_hidden = super(DDPG_Critic, self).__init__(D_in1, D_hidden, head)
        self.l1 = torch.nn.Linear(D_in1 + D_in2, D_hidden)
        self.l2 = torch.nn.Linear(D_hidden, D_hidden)
        self.q1 = torch.nn.Linear(D_hidden, 1)
        
    def forward(self, x1, x2):
        x1 = super(DDPG_Critic, self).forward(x1)
        x = torch.cat([x1, x2], dim=-1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.q1(x)
    
class DDPG_Actor(BaseNetwork):
    def __init__(self, D_in, D_out, D_hidden=512, head=None):
        D_in, D_hidden = super(DDPG_Actor, self).__init__(D_in, D_hidden, head)
        self.fc1 = torch.nn.Linear(D_in, D_hidden)
        self.fc2 = torch.nn.Linear(D_hidden, D_hidden)
        self.mu = torch.nn.Linear(D_hidden, D_out)

    def forward(self, x):
        x = super(DDPG_Actor, self).forward(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.mu(x))