import torch
import torch.nn.functional as F

from .base import BaseNetwork


class DDPG_Critic(BaseNetwork):
    def __init__(self, D_in1, D_in2, head="mlp", D_hidden=512):
        D_head_out = super(DDPG_Critic, self).__init__(D_in1, D_hidden, head)
        self.l = torch.nn.Linear(D_head_out + D_in2, D_hidden)
        self.q = torch.nn.Linear(D_hidden, 1)

    def forward(self, x1, x2):
        x1 = super(DDPG_Critic, self).forward(x1)
        x = torch.cat([x1, x2], dim=-1)
        x = F.relu(self.l(x))
        return self.q(x)


class DDPG_Actor(BaseNetwork):
    def __init__(self, D_in, D_out, D_hidden=512, head="mlp"):
        D_head_out = super(DDPG_Actor, self).__init__(D_in, D_hidden, head)
        self.mu = torch.nn.Linear(D_head_out, D_out)

    def forward(self, x):
        x = super(DDPG_Actor, self).forward(x)
        return torch.tanh(self.mu(x))
