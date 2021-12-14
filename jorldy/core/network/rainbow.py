import torch
import torch.nn.functional as F

from .base import BaseNetwork
from .utils import noisy_l, init_weights, orthogonal_init


class Rainbow(BaseNetwork):
    def __init__(
        self, D_in, D_out, N_atom, noise_type="factorized", D_hidden=512, head="mlp"
    ):
        D_head_out = super(Rainbow, self).__init__(D_in, D_hidden, head)
        self.D_out = D_out
        self.N_atom = N_atom
        self.noise_type = noise_type

        self.l = torch.nn.Linear(D_head_out, D_hidden)

        self.mu_w_a1, self.sig_w_a1, self.mu_b_a1, self.sig_b_a1 = init_weights(
            (D_hidden, D_hidden), noise_type
        )
        self.mu_w_v1, self.sig_w_v1, self.mu_b_v1, self.sig_b_v1 = init_weights(
            (D_hidden, D_hidden), noise_type
        )

        self.mu_w_a2, self.sig_w_a2, self.mu_b_a2, self.sig_b_a2 = init_weights(
            (D_hidden, N_atom * self.D_out), noise_type
        )
        self.mu_w_v2, self.sig_w_v2, self.mu_b_v2, self.sig_b_v2 = init_weights(
            (D_hidden, N_atom), noise_type
        )

        orthogonal_init(self.l)

    def forward(self, x, is_train):
        x = super(Rainbow, self).forward(x)

        x = F.relu(self.l(x))

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
        )  # [bs, num_action * N_atom]
        x_a = torch.reshape(
            x_a, (-1, self.D_out, self.N_atom)
        )  # [bs, num_action, N_atom]
        x_a_mean = x_a.mean(dim=1).unsqueeze(1)  # [bs, 1, N_atom]
        x_a = x_a - x_a_mean.repeat(1, self.D_out, 1)  # [bs, num_action, N_atom]

        # V stream : state value
        x_v = noisy_l(
            x_v,
            self.mu_w_v2,
            self.sig_w_v2,
            self.mu_b_v2,
            self.sig_b_v2,
            self.noise_type,
            is_train,
        )  # [bs, N_atom]
        x_v = torch.reshape(x_v, (-1, 1, self.N_atom))  # [bs, 1, N_atom]
        x_v = x_v.repeat(1, self.D_out, 1)  # [bs, num_action, N_atom]

        out = x_a + x_v  # [bs, num_action, N_atom]

        return out
