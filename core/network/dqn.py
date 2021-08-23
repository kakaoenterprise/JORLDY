import torch
import torch.nn.functional as F

from .base import BaseNetwork

class DQN(BaseNetwork):
    def __init__(self, D_in, D_out, D_hidden=512, head=None):
        D_in, D_hidden = super(DQN, self).__init__(D_in, D_hidden, head)

        self.l1 = torch.nn.Linear(D_in, D_hidden)
        self.l2 = torch.nn.Linear(D_hidden, D_hidden)
        self.q = torch.nn.Linear(D_hidden, D_out)

    def forward(self, x):
        x = super(DQN, self).forward(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.q(x)