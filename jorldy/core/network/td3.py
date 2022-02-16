import torch
import torch.nn.functional as F

from .base import BaseNetwork
from .utils import orthogonal_init


class TD3_Critic(torch.nn.Module):
    def __init__(self, D_in1, D_in2, D_hidden1=400, D_hidden2=300, **kwargs):
        super(TD3_Critic, self).__init__()
        self.q1_l1 = torch.nn.Linear(D_in1 + D_in2, D_hidden1)
        self.q1_l2 = torch.nn.Linear(D_hidden1, D_hidden2)
        self.q1_l3 = torch.nn.Linear(D_hidden2, 1)
        orthogonal_init(self.q1_l1)
        orthogonal_init(self.q1_l2)
        orthogonal_init(self.q1_l3, "linear")

        self.q2_l1 = torch.nn.Linear(D_in1 + D_in2, D_hidden1)
        self.q2_l2 = torch.nn.Linear(D_hidden1, D_hidden2)
        self.q2_l3 = torch.nn.Linear(D_hidden2, 1)
        orthogonal_init(self.q2_l1)
        orthogonal_init(self.q2_l2)
        orthogonal_init(self.q2_l3, "linear")

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)

        q1 = F.relu(self.q1_l1(x))
        q1 = F.relu(self.q1_l2(q1))
        q1 = self.q1_l3(q1)

        q2 = F.relu(self.q2_l1(x))
        q2 = F.relu(self.q2_l2(q2))
        q2 = self.q2_l3(q2)
        return q1, q2


class TD3_Actor(BaseNetwork):
    def __init__(self, D_in, D_out, D_hidden=400, D_hidden2=300, head="mlp"):
        D_head_out = super(TD3_Actor, self).__init__(D_in, D_hidden, head)
        self.l = torch.nn.Linear(D_head_out, D_hidden2)
        self.mu = torch.nn.Linear(D_hidden2, D_out)

        orthogonal_init(self.l)
        orthogonal_init(self.mu, "tanh")

    def forward(self, x):
        x = super(TD3_Actor, self).forward(x)
        x = F.relu(self.l(x))
        return torch.tanh(self.mu(x))
