import torch
import torch.nn.functional as F
import numpy as np

from .base import BaseNetwork
from .utils import noisy_l, init_weights, orthogonal_init


class RainbowIQN(BaseNetwork):
    def __init__(
        self, D_in, D_out, D_em, N_sample, noise_type, D_hidden=512, head="mlp"
    ):
        D_head_out = super(RainbowIQN, self).__init__(D_in, D_hidden, head)
        self.D_out = D_out
        self.noise_type = noise_type

        self.N_sample = N_sample
        self.i_pi = (torch.arange(0, D_em) * np.pi).view(1, 1, D_em)

        self.state_embed = torch.nn.Linear(D_head_out, D_hidden)
        self.sample_embed = torch.nn.Linear(D_em, D_hidden)

        self.l1 = torch.nn.Linear(D_hidden, D_hidden)
        self.l2 = torch.nn.Linear(D_hidden, D_hidden)

        self.mu_w_a1, self.sig_w_a1, self.mu_b_a1, self.sig_b_a1 = init_weights(
            (D_hidden, D_hidden), noise_type
        )
        self.mu_w_v1, self.sig_w_v1, self.mu_b_v1, self.sig_b_v1 = init_weights(
            (D_hidden, D_hidden), noise_type
        )

        self.mu_w_a2, self.sig_w_a2, self.mu_b_a2, self.sig_b_a2 = init_weights(
            (D_hidden, self.D_out), noise_type
        )
        self.mu_w_v2, self.sig_w_v2, self.mu_b_v2, self.sig_b_v2 = init_weights(
            (D_hidden, 1), noise_type
        )

        orthogonal_init([self.sample_embed, self.sample_embed, self.l1, self.l2])

    def forward(self, x, is_train, tau_min=0, tau_max=1):
        x = super(RainbowIQN, self).forward(x)
        state_embed = F.relu(self.state_embed(x))

        cos_term, tau = self.make_embed(x, tau_min, tau_max)
        tau_embed = F.relu(self.sample_embed(cos_term))

        embed = torch.unsqueeze(state_embed, 1) * tau_embed

        x = F.relu(self.l1(embed))
        x = F.relu(self.l2(x))

        x_a = F.relu(
            noisy_l(
                x,
                self.mu_w_a1,
                self.sig_w_a1,
                self.mu_b_a1,
                self.sig_b_a1,
                self.noise_type,
                is_train,
            )
        )
        x_v = F.relu(
            noisy_l(
                x,
                self.mu_w_v1,
                self.sig_w_v1,
                self.mu_b_v1,
                self.sig_b_v1,
                self.noise_type,
                is_train,
            )
        )

        # A stream : action advantage
        x_a = noisy_l(
            x_a,
            self.mu_w_a2,
            self.sig_w_a2,
            self.mu_b_a2,
            self.sig_b_a2,
            self.noise_type,
            is_train,
        )  # [bs, num_sample, num_action]
        x_a_mean = x_a.mean(dim=2).unsqueeze(2)  # [bs, num_sample, 1]
        x_a = x_a - x_a_mean.repeat(1, 1, self.D_out)  # [bs, num_sample, num_action]

        # V stream : state value
        x_v = noisy_l(
            x_v,
            self.mu_w_v2,
            self.sig_w_v2,
            self.mu_b_v2,
            self.sig_b_v2,
            self.noise_type,
            is_train,
        )  # [bs, num_sample, 1]
        x_v = x_v.repeat(1, 1, self.D_out)  # [bs, num_sample, num_action]

        out = x_a + x_v  # [bs, num_sample, num_action]

        return out, tau

    def make_embed(self, x, tau_min, tau_max):
        tau = (
            torch.FloatTensor(x.size(0), self.N_sample, 1)
            .uniform_(tau_min, tau_max)
            .to(x.device)
        )
        embed = torch.cos(tau * self.i_pi.to(x.device))
        return embed, tau
