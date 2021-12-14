import torch
import torch.nn.functional as F

from .base import BaseNetwork
from .utils import noisy_l, init_weights


class Noisy(BaseNetwork):
    def __init__(self, D_in, D_out, noise_type="factorized", D_hidden=512, head="mlp"):
        assert noise_type in ["independent", "factorized"]

        D_head_out = super(Noisy, self).__init__(D_in, D_hidden, head)
        self.noise_type = noise_type

        self.mu_w1, self.sig_w1, self.mu_b1, self.sig_b1 = init_weights(
            (D_head_out, D_hidden), noise_type
        )
        self.mu_w2, self.sig_w2, self.mu_b2, self.sig_b2 = init_weights(
            (D_hidden, D_out), noise_type
        )

    def forward(self, x, is_train):
        x = super(Noisy, self).forward(x)
        x = F.relu(
            noisy_l(
                x,
                self.mu_w1,
                self.sig_w1,
                self.mu_b1,
                self.sig_b1,
                self.noise_type,
                is_train,
            )
        )
        x = noisy_l(
            x,
            self.mu_w2,
            self.sig_w2,
            self.mu_b2,
            self.sig_b2,
            self.noise_type,
            is_train,
        )
        return x

    def get_sig_w_mean(self):
        sig_w_abs_mean1 = torch.abs(self.sig_w1).mean()
        sig_w_abs_mean2 = torch.abs(self.sig_w2).mean()

        return sig_w_abs_mean1, sig_w_abs_mean2
