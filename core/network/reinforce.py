import torch
import torch.nn.functional as F

class ContinuousPolicy(torch.nn.Module):
    def __init__(self, D_in, D_out, D_hidden=512):
        super(ContinuousPolicy, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        
        self.l1 = torch.nn.Linear(D_in, D_hidden)
        self.l2 = torch.nn.Linear(D_hidden, D_hidden)
        self.mu = torch.nn.Linear(D_hidden, D_out)
        self.log_std = torch.nn.Linear(D_hidden, D_out)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        
        mu = torch.clamp(self.mu(x), min=-8., max=8.)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-5., max=2.)
        return mu, log_std.exp()
    
    
class DiscretePolicy(torch.nn.Module):
    def __init__(self, D_in, D_out, D_hidden=512):
        super(DiscretePolicy, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        
        self.l1 = torch.nn.Linear(D_in, D_hidden)
        self.l2 = torch.nn.Linear(D_hidden, D_hidden)
        self.l3 = torch.nn.Linear(D_hidden, D_out)
        
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return F.softmax(self.l3(x), dim=-1)