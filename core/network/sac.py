import torch
import torch.nn.functional as F
        
from .base import BaseNetwork

class SAC_Critic(BaseNetwork):
    def __init__(self, D_in1, D_in2, head='mlp', D_hidden=512):
        D_head_out = super(SAC_Critic, self).__init__(D_in1, D_hidden, head)

        self.l = torch.nn.Linear(D_head_out + D_in2, D_hidden)
        self.q1 = torch.nn.Linear(D_hidden, 1)

        self.l_ = torch.nn.Linear(D_head_out + D_in2, D_hidden)
        self.q2 = torch.nn.Linear(D_hidden, 1)
        
    def forward(self, x1, x2):
        x1 = super(SAC_Critic, self).forward(x1)
        x = torch.cat([x1, x2], dim=-1)
        x, x_ = F.relu(self.l(x)), F.relu(self.l_(x))
        return self.q1(x), self.q2(x_)