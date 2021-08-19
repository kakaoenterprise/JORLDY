import torch
import torch.nn.functional as F

from .utils import header_dict

class BaseNetwork(torch.nn.Module):
    def __init__(self, D_in, D_hidden, header):
        super(BaseNetwork, self).__init__()
        self.header = None
        if header is not None:
            assert isinstance(header, str)
            self.header = header_dict[header](D_in, D_hidden, D_hidden)
            D_in = D_hidden
        return D_in, D_hidden
    
    def forward(self, x):
        x = x if self.header is None else self.header(x)
        return x