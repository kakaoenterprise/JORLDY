import torch
import torch.nn.functional as F

class SACActor(torch.nn.Module):
    def __init__(self, D_in, D_out, D_hidden=512):
        super(SACActor, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        
        self.l1 = torch.nn.Linear(D_in, D_hidden)
        self.l2 = torch.nn.Linear(D_hidden, D_hidden)
        self.mu = torch.nn.Linear(D_hidden, D_out)
        self.log_std = torch.nn.Linear(D_hidden, D_out)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.mu(x), self.log_std(x).exp()
        
class SACCritic(torch.nn.Module):
    def __init__(self, D_in, D_out, D_hidden=512):
        super(SACCritic, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        
        self.l1 = torch.nn.Linear(D_in, D_hidden)
        self.l2 = torch.nn.Linear(D_hidden, D_hidden)
        self.q1 = torch.nn.Linear(D_hidden, D_out)

        self.l1_ = torch.nn.Linear(D_in, D_hidden)
        self.l2_ = torch.nn.Linear(D_hidden, D_hidden)
        self.q2 = torch.nn.Linear(D_hidden, D_out)
        
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        
        x, x_ = F.relu(self.l1(x)), F.relu(self.l1_(x))
        x, x_ = F.relu(self.l2(x)), F.relu(self.l2_(x_))
        return self.q1(x), self.q2(x_)