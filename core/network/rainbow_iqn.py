import torch
import torch.nn.functional as F
import numpy as np 
from .noisy import noisy_l, init_weights

class Rainbow_IQN(torch.nn.Module):
    def __init__(self, D_in, D_out, D_em, N_sample, D_hidden=128):
        super(Rainbow_IQN, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.D_hidden = D_hidden
        
        self.D_em = D_em 
        self.N_sample = N_sample
        self.i_pi = (torch.arange(0, self.D_em) * np.pi).view(1, 1, self.D_em)
        
        self.state_embed = torch.nn.Linear(self.D_in, D_hidden)
        self.sample_embed = torch.nn.Linear(self.D_em, D_hidden)
        
        self.l1 = torch.nn.Linear(self.D_hidden, self.D_hidden)
        self.l2 = torch.nn.Linear(self.D_hidden, self.D_hidden)
        
        self.mu_w_a1, self.sig_w_a1, self.mu_b_a1, self.sig_b_a1 = init_weights((self.D_hidden, self.D_hidden))
        self.mu_w_v1, self.sig_w_v1, self.mu_b_v1, self.sig_b_v1 = init_weights((self.D_hidden, self.D_hidden))
        
        self.mu_w_a2, self.sig_w_a2, self.mu_b_a2, self.sig_b_a2 = init_weights((self.D_hidden, self.D_out))
        self.mu_w_v2, self.sig_w_v2, self.mu_b_v2, self.sig_b_v2 = init_weights((self.D_hidden, 1))
        
    def forward(self, x, is_train, tau_min=0, tau_max=1):
        state_embed = F.relu(self.state_embed(x))

        cos_term, tau = self.make_embed(x, tau_min, tau_max)
        tau_embed = F.relu(self.sample_embed(cos_term))

        embed = torch.unsqueeze(state_embed, 1) * tau_embed
        
        x = F.relu(self.l1(embed))
        x = F.relu(self.l2(x))
        
        x_a = F.relu(noisy_l(x, self.mu_w_a1, self.sig_w_a1, self.mu_b_a1, self.sig_b_a1, is_train))
        x_v = F.relu(noisy_l(x, self.mu_w_v1, self.sig_w_v1, self.mu_b_v1, self.sig_b_v1, is_train))
        
        # A stream : action advantage
        x_a = noisy_l(x_a, self.mu_w_a2, self.sig_w_a2, self.mu_b_a2, self.sig_b_a2, is_train) # [bs, num_sample, num_action]
        x_a_mean = x_a.mean(dim=2).unsqueeze(2) # [bs, num_sample, 1]
        x_a = x_a - x_a_mean.repeat(1, 1, self.D_out) # [bs, num_sample, num_action]

        # V stream : state value
        x_v = noisy_l(x_v, self.mu_w_v2, self.sig_w_v2, self.mu_b_v2, self.sig_b_v2, is_train) # [bs, num_sample, 1]
        x_v = x_v.repeat(1, 1, self.D_out) # [bs, num_sample, num_action]
        
        out = x_a + x_v # [bs, num_sample, num_action]
        
        return out, tau
    
    def make_embed(self, x, tau_min, tau_max):
        tau = torch.FloatTensor(x.size(0), self.N_sample, 1).uniform_(tau_min, tau_max).to(x.device)
        embed = torch.cos(tau * self.i_pi.to(x.device))
        return embed, tau
        
    
class Rainbow_IQN_CNN(torch.nn.Module):
    def __init__(self, D_in, D_out, D_em, N_sample):
        super(Rainbow_IQN_CNN, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        
        self.D_em = D_em 
        self.N_sample = N_sample
        self.i_pi = (torch.arange(0, self.D_em) * np.pi).view(1, 1, self.D_em)        
        
        self.conv1 = torch.nn.Conv2d(in_channels=self.D_in[0], out_channels=32, kernel_size=8, stride=4)
        dim1 = ((self.D_in[1] - 8)//4 + 1, (self.D_in[2] - 8)//4 + 1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        dim2 = ((dim1[0] - 4)//2 + 1, (dim1[1] - 4)//2 + 1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        dim3 = ((dim2[0] - 3)//1 + 1, (dim2[1] - 3)//1 + 1)
        
        self.sample_embed = torch.nn.Linear(D_em, 64*dim3[0]*dim3[1])
        
        self.l1 = torch.nn.Linear(64*dim3[0]*dim3[1], 512)
        
        self.mu_w_a1, self.sig_w_a1, self.mu_b_a1, self.sig_b_a1 = init_weights((512, 512))
        self.mu_w_v1, self.sig_w_v1, self.mu_b_v1, self.sig_b_v1 = init_weights((512, 512))
        
        self.mu_w_a2, self.sig_w_a2, self.mu_b_a2, self.sig_b_a2 = init_weights((512, self.D_out))
        self.mu_w_v2, self.sig_w_v2, self.mu_b_v2, self.sig_b_v2 = init_weights((512, 1))
        
    def forward(self, x, is_train, tau_min=0, tau_max=1):
        x = (x-(255.0/2))/(255.0/2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        state_embed = x.view(x.size(0), -1)
        
        cos_term, tau = self.make_embed(x, tau_min, tau_max)
        tau_embed = F.relu(self.sample_embed(cos_term))

        embed = torch.unsqueeze(state_embed, 1) * tau_embed
        
        x = F.relu(self.l1(embed))
        
        x_a = F.relu(noisy_l(x, self.mu_w_a1, self.sig_w_a1, self.mu_b_a1, self.sig_b_a1, is_train))
        x_v = F.relu(noisy_l(x, self.mu_w_v1, self.sig_w_v1, self.mu_b_v1, self.sig_b_v1, is_train))
        
        # A stream : action advantage
        x_a = noisy_l(x_a, self.mu_w_a2, self.sig_w_a2, self.mu_b_a2, self.sig_b_a2, is_train) # [bs, num_sample, num_action]
        x_a_mean = x_a.mean(dim=2).unsqueeze(2) # [bs, num_sample, 1]
        x_a = x_a - x_a_mean.repeat(1, 1, self.D_out) # [bs, num_sample, num_action]

        # V stream : state value
        x_v = noisy_l(x_v, self.mu_w_v2, self.sig_w_v2, self.mu_b_v2, self.sig_b_v2, is_train) # [bs, num_sample, 1]
        x_v = x_v.repeat(1, 1, self.D_out) # [bs, num_sample, num_action]
        
        out = x_a + x_v # [bs, num_action]
        
        return out, tau
    
    def make_embed(self, x, tau_min, tau_max):
        tau = torch.FloatTensor(x.size(0), self.N_sample, 1).uniform_(tau_min, tau_max).to(x.device)
        embed = torch.cos(tau * self.i_pi.to(x.device))
        return embed, tau