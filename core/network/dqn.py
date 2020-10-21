import torch
import torch.nn.functional as F

class DQN(torch.nn.Module):
    def __init__(self, D_in, D_out, D_hidden=512):
        super(DQN, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        
        self.l1 = torch.nn.Linear(D_in, D_hidden)
        self.l2 = torch.nn.Linear(D_hidden, D_hidden)
        self.q = torch.nn.Linear(D_hidden, D_out)

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.q(x)