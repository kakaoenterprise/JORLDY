import numpy as np

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
