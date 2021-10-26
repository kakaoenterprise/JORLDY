import torch
import torch.nn.functional as F

from .base import BaseNetwork

class ContinuousPolicy(BaseNetwork):
    def __init__(self, D_in, D_out, D_hidden=512, head='mlp'):
        D_head_out = super(ContinuousPolicy, self).__init__(D_in, D_hidden, head)
        self.l = torch.nn.Linear(D_head_out, D_hidden)
        self.mu = torch.nn.Linear(D_hidden, D_out)
        self.log_std = torch.nn.Linear(D_hidden, D_out)

    def forward(self, x):
        x = super(ContinuousPolicy, self).forward(x)
        x = F.relu(self.l(x))
        
        mu = torch.clamp(self.mu(x), min=-5., max=5.)
        log_std = torch.tanh(self.log_std(x))
        return mu, log_std.exp()
    
    
class DiscretePolicy(BaseNetwork):
    def __init__(self, D_in, D_out, D_hidden=512, head='mlp'):
        D_head_out = super(DiscretePolicy, self).__init__(D_in, D_hidden, head)
        self.l = torch.nn.Linear(D_head_out, D_hidden)
        self.pi = torch.nn.Linear(D_hidden, D_out)
        
    def forward(self, x):
        x = super(DiscretePolicy, self).forward(x)
        x = F.relu(self.l(x))
        return F.softmax(self.pi(x), dim=-1)