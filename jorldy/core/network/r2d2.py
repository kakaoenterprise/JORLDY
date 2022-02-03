import torch
import torch.nn.functional as F

from .base import BaseNetwork
from .utils import orthogonal_init


class R2D2(BaseNetwork):
    def __init__(self, D_in, D_out, D_hidden=512, head="mlp"):
        D_head_out = super(R2D2, self).__init__(D_in, D_hidden, head)
        self.D_hidden = D_hidden

        self.lstm = torch.nn.LSTM(
            input_size=D_head_out + D_out, hidden_size=D_hidden, batch_first=True
        )

        self.l = torch.nn.Linear(D_hidden, D_hidden)

        self.l1_a = torch.nn.Linear(D_hidden, D_hidden)
        self.l1_v = torch.nn.Linear(D_hidden, D_hidden)

        self.l2_a = torch.nn.Linear(D_hidden, D_out)
        self.l2_v = torch.nn.Linear(D_hidden, 1)

        orthogonal_init([self.l1_a, self.l1_v])
        orthogonal_init([self.l2_a, self.l2_v], "linear")

    def forward(self, x1, x2, hidden_in=None):
        x1 = super(R2D2, self).forward(x1)
        x = torch.cat([x1, x2], dim=-1)

        if hidden_in is None:
            hidden_in = (
                torch.zeros(1, x.size(0), self.D_hidden).to(x.device),
                torch.zeros(1, x.size(0), self.D_hidden).to(x.device),
            )

        x, hidden_out = self.lstm(x, hidden_in)

        x = F.relu(self.l(x))

        x_a = F.relu(self.l1_a(x))
        x_v = F.relu(self.l1_v(x))

        # A stream : action advantage
        x_a = self.l2_a(x_a)  # [bs, seq, num_action]
        x_a -= x_a.mean(dim=2, keepdim=True)  # [bs, seq, num_action]

        # V stream : state value
        x_v = self.l2_v(x_v)  # [bs, seq, 1]

        out = x_a + x_v  # [bs, seq, num_action]
        return out, hidden_in, hidden_out
