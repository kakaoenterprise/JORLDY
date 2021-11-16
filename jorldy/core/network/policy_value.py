import torch
import torch.nn.functional as F

from .base import BaseNetwork


class ContinuousPolicyValue(BaseNetwork):
    def __init__(self, D_in, D_out, D_hidden=512, head="mlp"):
        D_head_out = super(ContinuousPolicyValue, self).__init__(D_in, D_hidden, head)
        self.l = torch.nn.Linear(D_head_out, D_hidden)
        self.mu = torch.nn.Linear(D_hidden, D_out)
        self.log_std = torch.nn.Linear(D_hidden, D_out)
        self.v = torch.nn.Linear(D_hidden, 1)

    def forward(self, x):
        x = super(ContinuousPolicyValue, self).forward(x)
        x = F.relu(self.l(x))

        mu = torch.clamp(self.mu(x), min=-5.0, max=5.0)
        log_std = torch.tanh(self.log_std(x))
        return mu, log_std.exp(), self.v(x)


class ContinuousPolicySeparateValue(ContinuousPolicyValue):
    def __init__(self, D_in, D_out, D_hidden=512, head="mlp"):
        super(ContinuousPolicySeparateValue, self).__init__(D_in, D_out, D_hidden, head)
        self.v_i = torch.nn.Linear(D_hidden, 1)

    def get_vi(self, x):
        x = super(ContinuousPolicyValue, self).forward(x)
        x = F.relu(self.l(x))
        return self.v_i(x)


class DiscretePolicyValue(BaseNetwork):
    def __init__(self, D_in, D_out, D_hidden=512, head="mlp"):
        D_head_out = super(DiscretePolicyValue, self).__init__(D_in, D_hidden, head)
        self.l = torch.nn.Linear(D_head_out, D_hidden)
        self.pi = torch.nn.Linear(D_hidden, D_out)
        self.v = torch.nn.Linear(D_hidden, 1)

    def forward(self, x):
        x = super(DiscretePolicyValue, self).forward(x)
        x = F.relu(self.l(x))
        return F.softmax(self.pi(x), dim=-1), self.v(x)


class DiscretePolicySeparateValue(DiscretePolicyValue):
    def __init__(self, D_in, D_out, D_hidden=512, head="mlp"):
        super(DiscretePolicySeparateValue, self).__init__(D_in, D_out, D_hidden, head)
        self.v_i = torch.nn.Linear(D_hidden, 1)

    def get_vi(self, x):
        x = super(DiscretePolicyValue, self).forward(x)
        x = F.relu(self.l(x))
        return self.v_i(x)
