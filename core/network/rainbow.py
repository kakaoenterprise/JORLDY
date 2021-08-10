import torch
import torch.nn.functional as F
import numpy as np 

from .noisy import noisy_l, init_weights

class Rainbow(torch.nn.Module):
    def __init__(self, D_in, D_out, N_atom, D_hidden=512):
        super(Rainbow, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.N_atom = N_atom
        self.D_hidden = D_hidden
        
        self.l1 = torch.nn.Linear(D_in, D_hidden)
        self.l2 = torch.nn.Linear(D_hidden, D_hidden)
        
        self.mu_w_a1, self.sig_w_a1, self.mu_b_a1, self.sig_b_a1 = init_weights((self.D_hidden, self.D_hidden))
        self.mu_w_v1, self.sig_w_v1, self.mu_b_v1, self.sig_b_v1 = init_weights((self.D_hidden, self.D_hidden))
        
        self.mu_w_a2, self.sig_w_a2, self.mu_b_a2, self.sig_b_a2 = init_weights((self.D_hidden, self.N_atom * self.D_out))
        self.mu_w_v2, self.sig_w_v2, self.mu_b_v2, self.sig_b_v2 = init_weights((self.D_hidden, self.N_atom))
        
    def forward(self, x, is_train):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        
        x_a = F.relu(noisy_l(x, self.mu_w_a1, self.sig_w_a1, self.mu_b_a1, self.sig_b_a1, is_train))
        x_v = F.relu(noisy_l(x, self.mu_w_v1, self.sig_w_v1, self.mu_b_v1, self.sig_b_v1, is_train))
        
        # A stream : action advantage
        x_a = noisy_l(x_a, self.mu_w_a2, self.sig_w_a2, self.mu_b_a2, self.sig_b_a2, is_train) # [bs, num_action * N_atom]
        x_a = torch.reshape(x_a, (-1, self.D_out, self.N_atom)) # [bs, num_action, N_atom]
        x_a_mean = x_a.mean(dim=1).unsqueeze(1) # [bs, 1, N_atom]
        x_a = x_a - x_a_mean.repeat(1, self.D_out, 1) # [bs, num_action, N_atom]

        # V stream : state value
        x_v = noisy_l(x_v, self.mu_w_v2, self.sig_w_v2, self.mu_b_v2, self.sig_b_v2, is_train) # [bs, N_atom]
        x_v = torch.reshape(x_v, (-1, 1, self.N_atom)) # [bs, 1, N_atom]
        x_v = x_v.repeat(1, self.D_out, 1) # [bs, num_action, N_atom]
        
        out = x_a + x_v # [bs, num_action, N_atom]
        
        return out
    
class Rainbow_CNN(torch.nn.Module):
    def __init__(self, D_in, D_out, N_atom, D_hidden=512):
        super(Rainbow_CNN, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.N_atom = N_atom
        
        self.conv1 = torch.nn.Conv2d(in_channels=self.D_in[0], out_channels=32, kernel_size=8, stride=4)
        dim1 = ((self.D_in[1] - 8)//4 + 1, (self.D_in[2] - 8)//4 + 1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        dim2 = ((dim1[0] - 4)//2 + 1, (dim1[1] - 4)//2 + 1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        dim3 = ((dim2[0] - 3)//1 + 1, (dim2[1] - 3)//1 + 1)
        
        self.l1 = torch.nn.Linear(64*dim3[0]*dim3[1], D_hidden)
        
        self.mu_w_a1, self.sig_w_a1, self.mu_b_a1, self.sig_b_a1 = init_weights((D_hidden, D_hidden))
        self.mu_w_v1, self.sig_w_v1, self.mu_b_v1, self.sig_b_v1 = init_weights((D_hidden, D_hidden))
        
        self.mu_w_a2, self.sig_w_a2, self.mu_b_a2, self.sig_b_a2 = init_weights((D_hidden, self.N_atom * self.D_out))
        self.mu_w_v2, self.sig_w_v2, self.mu_b_v2, self.sig_b_v2 = init_weights((D_hidden, self.N_atom))
        
    def forward(self, x, is_train):
        x = (x-(255.0/2))/(255.0/2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.l1(x))
        
        x_a = F.relu(noisy_l(x, self.mu_w_a1, self.sig_w_a1, self.mu_b_a1, self.sig_b_a1, is_train))
        x_v = F.relu(noisy_l(x, self.mu_w_v1, self.sig_w_v1, self.mu_b_v1, self.sig_b_v1, is_train))
        
        # A stream : action advantage
        x_a = noisy_l(x_a, self.mu_w_a2, self.sig_w_a2, self.mu_b_a2, self.sig_b_a2, is_train) # [bs, num_action * N_atom]
        x_a = torch.reshape(x_a, (-1, self.D_out, self.N_atom)) # [bs, num_action, N_atom]
        x_a_mean = x_a.mean(dim=1).unsqueeze(1) # [bs, 1, N_atom]
        x_a = x_a - x_a_mean.repeat(1, self.D_out, 1) # [bs, num_action, N_atom]

        # V stream : state value
        x_v = noisy_l(x_v, self.mu_w_v2, self.sig_w_v2, self.mu_b_v2, self.sig_b_v2, is_train) # [bs, N_atom]
        x_v = torch.reshape(x_v, (-1, 1, self.N_atom)) # [bs, 1, N_atom]
        x_v = x_v.repeat(1, self.D_out, 1) # [bs, num_action, N_atom]
        
        out = x_a + x_v # [bs, num_action, N_atom]
        
        return out