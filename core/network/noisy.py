import torch
import torch.nn.functional as F

from .base import BaseNetwork

def noisy_l(x, mu_w, sig_w, mu_b, sig_b, is_train):          
#     # Independent Gaussian Noise
#     if is_train:
#         eps_w = torch.randn(mu_w.size()).to(x.device)
#         eps_b = torch.randn(mu_b.size()).to(x.device)
#     else:
#         eps_w = torch.zeros(mu_w.size()).to(x.device)
#         eps_b = torch.zeros(mu_b.size()).to(x.device)

    # Factorized Gaussian Noise
    if is_train:
        eps_i = torch.randn(mu_w.size()[0]).to(x.device)
        eps_j = torch.randn(mu_b.size()[0]).to(x.device)

        f_eps_i = torch.sign(eps_i) * torch.sqrt(torch.abs(eps_i))
        f_eps_j = torch.sign(eps_j) * torch.sqrt(torch.abs(eps_j))

        eps_w = torch.matmul(torch.unsqueeze(f_eps_i, 1), torch.unsqueeze(f_eps_j, 0))
        eps_b = f_eps_j
    else:
        eps_i = torch.zeros(mu_w.size()[0],1).to(x.device)
        eps_j = torch.zeros(1,mu_b.size()[0]).to(x.device)

        eps_w = torch.matmul(eps_i, eps_j)
        eps_b = eps_j

    weight = mu_w + sig_w * eps_w
    bias = mu_b + sig_b * eps_b

    y = torch.matmul(x, weight) + bias

    return y

def init_weights(shape):
    # Independent: mu: sqrt(3/shape[0]), sig: 0.017
    # Factorised: mu: sqrt(1/shape[0]), sig: 0.4/sqrt(shape[0])
    
    mu_init = 1./(shape[0]**0.5)
    sig_init = 0.4/(shape[0]**0.5)
    
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
    def __init__(self, D_in, D_out, D_hidden=512, header=None):
        D_in, D_hidden = super(Noisy, self).__init__(D_in, D_hidden, header)

        self.mu_w, self.sig_w, self.mu_b, self.sig_b = init_weights((D_hidden, D_out))
        self.l1 = torch.nn.Linear(D_in, D_hidden)
        self.l2 = torch.nn.Linear(D_hidden, D_hidden)
    
    def forward(self, x, is_train):
        x = super(Noisy, self).forward(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
    
        return noisy_l(x, self.mu_w, self.sig_w, self.mu_b, self.sig_b, is_train)