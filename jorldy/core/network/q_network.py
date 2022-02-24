import torch
import torch.nn.functional as F

from .base import BaseNetwork
from .utils import orthogonal_init


class DiscreteQ_Network(BaseNetwork):
    def __init__(self, D_in, D_out, D_hidden=512, head="mlp"):
        D_head_out = super(DiscreteQ_Network, self).__init__(D_in, D_hidden, head)
        self.l = torch.nn.Linear(D_head_out, D_hidden)
        self.q = torch.nn.Linear(D_hidden, D_out)

        orthogonal_init(self.l)
        orthogonal_init(self.q, "linear")

    def forward(self, x):
        x = super(DiscreteQ_Network, self).forward(x)
        x = F.relu(self.l(x))
        return self.q(x)


class ContinuousQ_Network(BaseNetwork):
    def __init__(self, D_in1, D_in2, head="mlp", D_hidden=512):
        D_head_out = super(ContinuousQ_Network, self).__init__(D_in1, D_hidden, head)
        self.e = torch.nn.Linear(D_in2, D_hidden)
        self.l = torch.nn.Linear(D_hidden + D_head_out, D_hidden)
        self.q = torch.nn.Linear(D_hidden, 1)

        orthogonal_init(self.e)
        orthogonal_init(self.l)
        orthogonal_init(self.q, "linear")

    def forward(self, x1, x2):
        x1 = super(ContinuousQ_Network, self).forward(x1)
        x2 = F.relu(self.e(x2))
        x = torch.cat([x1, x2], dim=-1)
        x = F.relu(self.l(x))
        return self.q(x)
