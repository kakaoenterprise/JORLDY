import torch
import torch.nn.functional as F

from .base import BaseNetwork
from .utils import orthogonal_init


class DeterministicPolicy(BaseNetwork):
    def __init__(self, D_in, D_out, D_hidden=512, head="mlp"):
        D_head_out = super(DeterministicPolicy, self).__init__(D_in, D_hidden, head)
        self.l = torch.nn.Linear(D_head_out, D_hidden)
        self.pi = torch.nn.Linear(D_hidden, D_out)

        orthogonal_init(self.l)
        orthogonal_init(self.pi, "tanh")

    def forward(self, x):
        x = super(DeterministicPolicy, self).forward(x)
        x = F.relu(self.l(x))
        return torch.tanh(self.pi(x))


class DiscretePolicy(BaseNetwork):
    def __init__(self, D_in, D_out, D_hidden=512, head="mlp"):
        D_head_out = super(DiscretePolicy, self).__init__(D_in, D_hidden, head)
        self.l = torch.nn.Linear(D_head_out, D_hidden)
        self.pi = torch.nn.Linear(D_hidden, D_out)

        orthogonal_init(self.l)
        orthogonal_init(self.pi, "policy")

    def forward(self, x):
        x = super(DiscretePolicy, self).forward(x)
        x = F.relu(self.l(x))
        return torch.exp(F.log_softmax(self.pi(x), dim=-1))


class ContinuousPolicy(BaseNetwork):
    def __init__(self, D_in, D_out, D_hidden=512, head="mlp"):
        D_head_out = super(ContinuousPolicy, self).__init__(D_in, D_hidden, head)
        self.l = torch.nn.Linear(D_head_out, D_hidden)
        self.mu = torch.nn.Linear(D_hidden, D_out)
        self.log_std = torch.nn.Linear(D_hidden, D_out)

        orthogonal_init(self.l)
        orthogonal_init(self.mu, "linear")
        orthogonal_init(self.log_std, "tanh")

    def forward(self, x):
        x = super(ContinuousPolicy, self).forward(x)
        x = F.relu(self.l(x))

        mu = torch.clamp(self.mu(x), min=-5.0, max=5.0)
        log_std = torch.tanh(self.log_std(x))
        return mu, log_std.exp()
