import torch
import torch.nn.functional as F
        
from .base import BaseNetwork

class SAC_Critic(BaseNetwork):
    def __init__(self, D_in, D_out, head=None, D_hidden=512):
        D_in, D_hidden = super(SAC_Critic, self).__init__(D_in, D_hidden, head)

        self.l1 = torch.nn.Linear(D_in, D_hidden)
        self.l2 = torch.nn.Linear(D_hidden, D_hidden)
        self.q1 = torch.nn.Linear(D_hidden, D_out)

        self.l1_ = torch.nn.Linear(D_in, D_hidden)
        self.l2_ = torch.nn.Linear(D_hidden, D_hidden)
        self.q2 = torch.nn.Linear(D_hidden, D_out)
        
    def forward(self, x1, x2):
        x = super(SAC_Critic, self).forward(x1)
        x = torch.cat([x1, x2], dim=-1)
        x, x_ = F.relu(self.l1(x)), F.relu(self.l1_(x))
        x, x_ = F.relu(self.l2(x)), F.relu(self.l2_(x_))
        return self.q1(x), self.q2(x_)