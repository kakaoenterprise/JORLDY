import torch
import torch.nn.functional as F

from .base import BaseNetwork


class DQN(BaseNetwork):
    def __init__(self, D_in, D_out, D_hidden=512, head="mlp"):
        D_head_out = super(DQN, self).__init__(D_in, D_hidden, head)
        self.l = torch.nn.Linear(D_head_out, D_hidden)
        self.q = torch.nn.Linear(D_hidden, D_out)

    def forward(self, x):
        x = super(DQN, self).forward(x)
        x = F.relu(self.l(x))
        return self.q(x)
