import torch
import torch.nn.functional as F
import numpy as np 

class Rainbow(torch.nn.Module):
    def __init__(self, D_in, D_out, num_atom, device, D_hidden=512):
        super(Rainbow, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.Num_atom = num_atom
        self.D_hidden = D_hidden
        self.device = device
        
        self.l1 = torch.nn.Linear(D_in, D_hidden)
        self.l2 = torch.nn.Linear(D_hidden, D_hidden)
        
        self.mu_w_a1, self.sig_w_a1, self.mu_b_a1, self.sig_b_a1 = self.init_weights((self.D_hidden, self.D_hidden))
        self.mu_w_v1, self.sig_w_v1, self.mu_b_v1, self.sig_b_v1 = self.init_weights((self.D_hidden, self.D_hidden))
        
        self.mu_w_a2, self.sig_w_a2, self.mu_b_a2, self.sig_b_a2 = self.init_weights((self.D_hidden, self.Num_atom * self.D_out))
        self.mu_w_v2, self.sig_w_v2, self.mu_b_v2, self.sig_b_v2 = self.init_weights((self.D_hidden, self.Num_atom))
        
    def forward(self, x, is_train):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        
        x_a = F.relu(self.noisy_l(x, self.mu_w_a1, self.sig_w_a1, self.mu_b_a1, self.sig_b_a1, is_train))
        x_v = F.relu(self.noisy_l(x, self.mu_w_v1, self.sig_w_v1, self.mu_b_v1, self.sig_b_v1, is_train))
        
        # A stream : action advantage
        x_a = self.noisy_l(x_a, self.mu_w_a2, self.sig_w_a2, self.mu_b_a2, self.sig_b_a2, is_train) # [bs, num_action * num_atom]
        x_a = torch.reshape(x_a, (-1, self.D_out, self.Num_atom)) # [bs, num_action, num_atom]
        x_a_mean = x_a.mean(dim=1).unsqueeze(1) # [bs, 1, num_atom]
        x_a = x_a - x_a_mean.repeat(1, self.D_out, 1) # [bs, num_action, num_atom]

        # V stream : state value
        x_v = self.noisy_l(x_v, self.mu_w_v2, self.sig_w_v2, self.mu_b_v2, self.sig_b_v2, is_train) # [bs, num_atom]
        x_v = torch.reshape(x_v, (-1, 1, self.Num_atom)) # [bs, num_action, num_atom]
        x_v = x_v.repeat(1, self.D_out, 1) # [bs, num_action, num_atom]
        
        out = x_a + x_v # [bs, num_action, num_atom]
        
        return out
    
    def noisy_l(self, x, mu_w, sig_w, mu_b, sig_b, is_train):          
        # Factorized Gaussian Noise
        if is_train:
            eps_i = torch.randn(mu_w.size()[0]).to(self.device)
            eps_j = torch.randn(mu_b.size()[0]).to(self.device)
            
            f_eps_i = torch.sign(eps_i) * torch.sqrt(torch.abs(eps_i))
            f_eps_j = torch.sign(eps_j) * torch.sqrt(torch.abs(eps_j))
            
            eps_w = torch.matmul(torch.unsqueeze(f_eps_i, 1), torch.unsqueeze(f_eps_j, 0))
            eps_b = f_eps_j
        else:
            eps_i = torch.zeros(mu_w.size()[0],1).to(self.device)
            eps_j = torch.zeros(1,mu_b.size()[0]).to(self.device)
            
            eps_w = torch.matmul(eps_i, eps_j)
            eps_b = eps_j

        weight = mu_w + sig_w * eps_w
        bias = mu_b + sig_b * eps_b

        y = torch.matmul(x, weight) + bias

        return y
            
    def init_weights(self, shape):
        # Independent: mu: np.sqrt(3/shape[0]), sig: 0.017
        # Factorised: mu: np.sqrt(1/shape[0]), sig: 0.4/np.sqrt(shape[0])
        mu_w = torch.nn.Parameter(torch.empty(shape[0],shape[1]))
        sig_w = torch.nn.Parameter(torch.full((shape[0],shape[1]), 0.5))
        mu_b = torch.nn.Parameter(torch.empty(shape[1]))
        sig_b = torch.nn.Parameter(torch.full((shape[1],), 0.5))
        
        mu_w.data.uniform_(-np.sqrt(1/shape[0]), np.sqrt(1/shape[0]))
        mu_b.data.uniform_(-np.sqrt(1/shape[0]), np.sqrt(1/shape[0]))
        
        return mu_w, sig_w, mu_b, sig_b
        
    
class Rainbow_CNN(torch.nn.Module):
    def __init__(self, D_in, D_out, Num_atom, device):
        super(Rainbow_CNN, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.Num_atom = Num_atom
        self.device = device 
        
        self.conv1 = torch.nn.Conv2d(in_channels=self.D_in[0], out_channels=32, kernel_size=8, stride=4)
        dim1 = ((self.D_in[1] - 8)//4 + 1, (self.D_in[2] - 8)//4 + 1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        dim2 = ((dim1[0] - 4)//2 + 1, (dim1[1] - 4)//2 + 1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        dim3 = ((dim2[0] - 3)//1 + 1, (dim2[1] - 3)//1 + 1)
        
        self.mu_w1, self.sig_w1, self.mu_b1, self.sig_b1 = self.init_weights((64*dim3[0]*dim3[1], 512))
        self.mu_w2, self.sig_w2, self.mu_b2, self.sig_b2 = self.init_weights((512, self.D_out))
        
        self.mu_w_a1, self.sig_w_a1, self.mu_b_a1, self.sig_b_a1 = self.init_weights((64*dim3[0]*dim3[1], 512))
        self.mu_w_v1, self.sig_w_v1, self.mu_b_v1, self.sig_b_v1 = self.init_weights((64*dim3[0]*dim3[1], 512))
        
        self.mu_w_a2, self.sig_w_a2, self.mu_b_a2, self.sig_b_a2 = self.init_weights((512, self.Num_atom * self.D_out))
        self.mu_w_v2, self.sig_w_v2, self.mu_b_v2, self.sig_b_v2 = self.init_weights((512, self.Num_atom))
        
    def forward(self, x, is_train):
        x = (x-(255.0/2))/(255.0/2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        x_a = F.relu(self.noisy_l(x, self.mu_w_a1, self.sig_w_a1, self.mu_b_a1, self.sig_b_a1, is_train))
        x_v = F.relu(self.noisy_l(x, self.mu_w_v1, self.sig_w_v1, self.mu_b_v1, self.sig_b_v1, is_train))
        
        # A stream : action advantage
        x_a = self.noisy_l(x_a, self.mu_w_a2, self.sig_w_a2, self.mu_b_a2, self.sig_b_a2, is_train) # [bs, num_action * num_atom]
        x_a = torch.reshape(x_a, (-1, self.D_out, self.Num_atom)) # [bs, num_action, num_atom]
        x_a_mean = x_a.mean(dim=1).unsqueeze(1) # [bs, 1, num_atom]
        x_a = x_a - x_a_mean.repeat(1, self.D_out, 1) # [bs, num_action, num_atom]

        # V stream : state value
        x_v = self.noisy_l(x_v, self.mu_w_v2, self.sig_w_v2, self.mu_b_v2, self.sig_b_v2, is_train) # [bs, num_atom]
        x_v = x_v.repeat(1, self.D_out, 1) # [bs, num_action, num_atom]
        
        return x

    def noisy_l(self, x, mu_w, sig_w, mu_b, sig_b, is_train):          
        # Factorized Gaussian Noise
        if is_train:
            eps_i = torch.randn(mu_w.size()[0]).to(self.device)
            eps_j = torch.randn(mu_b.size()[0]).to(self.device)
            
            f_eps_i = torch.sign(eps_i) * torch.sqrt(torch.abs(eps_i))
            f_eps_j = torch.sign(eps_j) * torch.sqrt(torch.abs(eps_j))
            
            eps_w = torch.matmul(torch.unsqueeze(f_eps_i, 1), torch.unsqueeze(f_eps_j, 0))
            eps_b = f_eps_j
        else:
            eps_i = torch.zeros(mu_w.size()[0],1).to(self.device)
            eps_j = torch.zeros(1,mu_b.size()[0]).to(self.device)
            
            eps_w = torch.matmul(eps_i, eps_j)
            eps_b = eps_j

        weight = mu_w + sig_w * eps_w
        bias = mu_b + sig_b * eps_b

        y = torch.matmul(x, weight) + bias

        return y
            
    def init_weights(self, shape):
        # Independent: mu: np.sqrt(3/shape[0]), sig: 0.017
        # Factorised: mu: np.sqrt(1/shape[0]), sig: 0.4/np.sqrt(shape[0])
        mu_w = torch.nn.Parameter(torch.empty(shape[0],shape[1]))
        sig_w = torch.nn.Parameter(torch.full((shape[0],shape[1]), 0.5))
        mu_b = torch.nn.Parameter(torch.empty(shape[1]))
        sig_b = torch.nn.Parameter(torch.full((shape[1],), 0.5))
        
        mu_w.data.uniform_(-np.sqrt(1/shape[0]), np.sqrt(1/shape[0]))
        mu_b.data.uniform_(-np.sqrt(1/shape[0]), np.sqrt(1/shape[0]))
        
        return mu_w, sig_w, mu_b, sig_b