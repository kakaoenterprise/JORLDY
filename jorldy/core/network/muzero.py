import torch
import torch.nn.functional as F

from .base import BaseNetwork
from .utils import orthogonal_init


class MuZero(BaseNetwork):
    def __init__(self, D_in, D_out, D_hidden=512, head="resnet"):
        D_head_out = super(MuZero, self).__init__(D_in, D_hidden, head)
        self.l = torch.nn.Linear(D_head_out, D_hidden)
        self.pi = torch.nn.Linear(D_hidden, D_out)

        orthogonal_init(self.l)
        orthogonal_init(self.pi, "tanh")

    def forward(self, x):
        pass
