import torch
import torch.nn.functional as F

from .base import BaseNetwork
from .utils import orthogonal_init


class Dueling(BaseNetwork):
    def __init__(self, D_in, D_out, D_hidden=512, head="mlp"):
        D_head_out = super(Dueling, self).__init__(D_in, D_hidden, head)

        self.l1_a = torch.nn.Linear(D_head_out, D_hidden)
        self.l1_v = torch.nn.Linear(D_head_out, D_hidden)

        self.l2_a = torch.nn.Linear(D_hidden, D_out)
        self.l2_v = torch.nn.Linear(D_hidden, 1)

        orthogonal_init([self.l1_a, self.l1_v])
        orthogonal_init([self.l2_a, self.l2_v], "linear")

    def forward(self, x):
        x = super(Dueling, self).forward(x)

        x_a = F.relu(self.l1_a(x))
        x_v = F.relu(self.l1_v(x))

        # A stream : action advantage
        x_a = self.l2_a(x_a)  # [bs, num_action]
        x_a -= x_a.mean(dim=1, keepdim=True)  # [bs, num_action]

        # V stream : state value
        x_v = self.l2_v(x_v)  # [bs, 1]

        out = x_a + x_v  # [bs, num_action]
        return out
