import numpy as np

import torch
import torch.nn.functional as F

# OU noise class
class OU_Noise:
    def __init__(self, action_size, mu, theta, sigma):
        self.action_size = action_size

        self.mu = mu
        self.theta = theta
        self.sigma = sigma

        self.reset()

    def reset(self):
        self.X = np.ones((1, self.action_size), dtype=np.float32) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X) + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


# Reference: m-rl official repository (stable_scaled_log_softmax, stable_softmax)
# https://github.com/google-research/google-research/blob/master/munchausen_rl/common/utils.py
def stable_scaled_log_softmax(x, tau):
    max_x, max_indices = torch.max(x, -1, keepdim=True)
    y = x - max_x
    tau_lse = max_x + tau * torch.log(torch.sum(torch.exp(y / tau), -1, keepdim=True))
    return x - tau_lse


def stable_softmax(x, tau):
    max_x, max_indices = torch.max(x, -1, keepdim=True)
    y = x - max_x
    return torch.exp(F.log_softmax(y / tau, -1))
