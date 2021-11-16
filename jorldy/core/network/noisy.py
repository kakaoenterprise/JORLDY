import torch
import torch.nn.functional as F

from .base import BaseNetwork


def noisy_l(x, mu_w, sig_w, mu_b, sig_b, noise_type, is_train):
    if noise_type == "factorized":
        # Factorized Gaussian Noise
        if is_train:
            eps_i = torch.randn(mu_w.size()[0]).to(x.device)
            eps_j = torch.randn(mu_b.size()[0]).to(x.device)

            f_eps_i = torch.sign(eps_i) * torch.sqrt(torch.abs(eps_i))
            f_eps_j = torch.sign(eps_j) * torch.sqrt(torch.abs(eps_j))

            eps_w = torch.matmul(
                torch.unsqueeze(f_eps_i, 1), torch.unsqueeze(f_eps_j, 0)
            )
            eps_b = f_eps_j
        else:
            eps_w = torch.zeros(mu_w.size()[0], mu_b.size()[0]).to(x.device)
            eps_b = torch.zeros(1, mu_b.size()[0]).to(x.device)
    else:
        # Independent Gaussian Noise
        if is_train:
            eps_w = torch.randn(mu_w.size()).to(x.device)
            eps_b = torch.randn(mu_b.size()).to(x.device)
        else:
            eps_w = torch.zeros(mu_w.size()).to(x.device)
            eps_b = torch.zeros(mu_b.size()).to(x.device)

    weight = mu_w + sig_w * eps_w
    bias = mu_b + sig_b * eps_b

    y = torch.matmul(x, weight) + bias

    return y


def init_weights(shape, noise_type):
    if noise_type == "factorized":
        mu_init = 1.0 / (shape[0] ** 0.5)
        sig_init = 0.5 / (shape[0] ** 0.5)
    else:
        mu_init = (3.0 / shape[0]) ** 0.5
        sig_init = 0.017

    mu_w = torch.nn.Parameter(torch.empty(shape))
    sig_w = torch.nn.Parameter(torch.empty(shape))
    mu_b = torch.nn.Parameter(torch.empty(shape[1]))
    sig_b = torch.nn.Parameter(torch.empty(shape[1]))

    mu_w.data.uniform_(-mu_init, mu_init)
    mu_b.data.uniform_(-mu_init, mu_init)
    sig_w.data.uniform_(sig_init, sig_init)
    sig_b.data.uniform_(sig_init, sig_init)

    return mu_w, sig_w, mu_b, sig_b


class Noisy(BaseNetwork):
    def __init__(self, D_in, D_out, noise_type="factorized", D_hidden=512, head="mlp"):
        D_head_out = super(Noisy, self).__init__(D_in, D_hidden, head)
        self.noise_type = noise_type

        # self.l = torch.nn.Linear(D_head_out, D_hidden)
        self.mu_w1, self.sig_w1, self.mu_b1, self.sig_b1 = init_weights(
            (D_head_out, D_hidden), noise_type
        )
        self.mu_w2, self.sig_w2, self.mu_b2, self.sig_b2 = init_weights(
            (D_hidden, D_out), noise_type
        )

    def forward(self, x, is_train):
        x = super(Noisy, self).forward(x)
        # x = F.relu(self.l(x))
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
