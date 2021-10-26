import torch
import torch.nn.functional as F

from .base import BaseNetwork

class R2D2(BaseNetwork):
    def __init__(self, D_in, D_out, D_hidden=512, head='mlp_lstm'):
        D_head_out = super(R2D2, self).__init__(D_in, D_hidden, head)
        
        self.l = torch.nn.Linear(D_head_out, D_hidden)

        self.l1_a = torch.nn.Linear(D_hidden, D_hidden)
        self.l1_v = torch.nn.Linear(D_hidden, D_hidden)

        self.l2_a = torch.nn.Linear(D_hidden, D_out)
        self.l2_v = torch.nn.Linear(D_hidden, 1)

    def forward(self, x, hidden_in=None):
        x, hidden_in, hidden_out = super(R2D2, self).forward(x, hidden_in)
        
        x = F.relu(self.l(x))
        
        x_a = F.relu(self.l1_a(x))
        x_v = F.relu(self.l1_v(x))
        
        # A stream : action advantage
        x_a = self.l2_a(x_a) # [bs, seq, num_action]
        x_a -= x_a.mean(dim=2, keepdim=True) # [bs, seq, num_action]

        # V stream : state value
        x_v = self.l2_v(x_v) # [bs, seq, 1]

        out = x_a + x_v # [bs, seq, num_action]
        return out, hidden_in, hidden_out
