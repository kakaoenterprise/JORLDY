import torch
import torch.nn.functional as F

from .utils import head_dict

class BaseNetwork(torch.nn.Module):
    def __init__(self, D_in, D_hidden, head):
        super(BaseNetwork, self).__init__()
        assert head in head_dict.keys()
        self.head = head_dict[head](D_in, D_hidden)
        return self.head.D_head_out
    
    def forward(self, x):
        x = self.head(x)
        return x
        