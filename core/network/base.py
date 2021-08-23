import torch
import torch.nn.functional as F

from .utils import head_dict

class BaseNetwork(torch.nn.Module):
    def __init__(self, D_in, D_hidden, head):
        super(BaseNetwork, self).__init__()
        self.head = lambda x : x
        if head is not None:
            assert isinstance(head, str)
            self.head = head_dict[head](D_in, D_hidden, D_hidden)
            D_in = D_hidden
        return D_in, D_hidden
    
    def forward(self, x):
        x = self.head(x)
        return x
        