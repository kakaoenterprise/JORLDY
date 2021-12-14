import torch
import torch.nn.functional as F
import numpy as np

from .base import BaseNetwork
from .utils import orthogonal_init


class IQN(BaseNetwork):
    def __init__(self, D_in, D_out, D_em=64, N_sample=64, D_hidden=512, head="mlp"):
        D_head_out = super(IQN, self).__init__(D_in, D_hidden, head)

        self.N_sample = N_sample
        self.i_pi = (torch.arange(0, D_em) * np.pi).view(1, 1, D_em)

        self.state_embed = torch.nn.Linear(D_head_out, D_hidden)
        self.sample_embed = torch.nn.Linear(D_em, D_hidden)

        self.l1 = torch.nn.Linear(D_hidden, D_hidden)
        self.l2 = torch.nn.Linear(D_hidden, D_hidden)
        self.q = torch.nn.Linear(D_hidden, D_out)

        orthogonal_init([self.sample_embed, self.sample_embed, self.l1, self.l2])
        orthogonal_init(self.q, "linear")

    def forward(self, x, tau_min=0, tau_max=1):
        assert 0 <= tau_min <= tau_max <= 1
        x = super(IQN, self).forward(x)
        state_embed = F.relu(self.state_embed(x))

        cos_term, tau = self.make_embed(x, tau_min, tau_max)
        tau_embed = F.relu(self.sample_embed(cos_term))

        embed = torch.unsqueeze(state_embed, 1) * tau_embed

        x = F.relu(self.l1(embed))
        x = F.relu(self.l2(x))
        return self.q(x), tau

    def make_embed(self, x, tau_min, tau_max):
        tau = (
            torch.FloatTensor(x.size(0), self.N_sample, 1)
            .uniform_(tau_min, tau_max)
            .to(x.device)
        )
        embed = torch.cos(tau * self.i_pi.to(x.device))
        return embed, tau
