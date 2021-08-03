import torch
import torch.nn.functional as F

def noisy_l(x, mu_w, sig_w, mu_b, sig_b, is_train):          
#     # Independent Gaussian Noise
#     if is_train:
#         eps_w = torch.randn(mu_w.size()).to(x.device)
#         eps_b = torch.randn(mu_b.size()).to(x.device)
#     else:
#         eps_w = torch.zeros(mu_w.size()).to(x.device)
#         eps_b = torch.zeros(mu_b.size()).to(x.device)

    # Factorized Gaussian Noise
    if is_train:
        eps_i = torch.randn(mu_w.size()[0]).to(x.device)
        eps_j = torch.randn(mu_b.size()[0]).to(x.device)

        f_eps_i = torch.sign(eps_i) * torch.sqrt(torch.abs(eps_i))
        f_eps_j = torch.sign(eps_j) * torch.sqrt(torch.abs(eps_j))

        eps_w = torch.matmul(torch.unsqueeze(f_eps_i, 1), torch.unsqueeze(f_eps_j, 0))
        eps_b = f_eps_j
    else:
        eps_i = torch.zeros(mu_w.size()[0],1).to(x.device)
        eps_j = torch.zeros(1,mu_b.size()[0]).to(x.device)

        eps_w = torch.matmul(eps_i, eps_j)
        eps_b = eps_j

    weight = mu_w + sig_w * eps_w
    bias = mu_b + sig_b * eps_b

    y = torch.matmul(x, weight) + bias

    return y

def init_weights(shape):
    # Independent: mu: sqrt(3/shape[0]), sig: 0.017
    # Factorised: mu: sqrt(1/shape[0]), sig: 0.4/sqrt(shape[0])
    
    mu_init = 1./(shape[0]**0.5)
    sig_init = 0.4/(shape[0]**0.5)
    
    mu_w = torch.nn.Parameter(torch.empty(shape))
    sig_w = torch.nn.Parameter(torch.empty(shape))
    mu_b = torch.nn.Parameter(torch.empty(shape[1]))
    sig_b = torch.nn.Parameter(torch.empty(shape[1]))

    mu_w.data.uniform_(-mu_init, mu_init)
    mu_b.data.uniform_(-mu_init, mu_init)
    sig_w.data.uniform_(sig_init, sig_init)
    sig_b.data.uniform_(sig_init, sig_init)
    
    return mu_w, sig_w, mu_b, sig_b
    
class Noisy(torch.nn.Module):
    def __init__(self, D_in, D_out, D_hidden=512):
        super(Noisy, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.D_hidden = D_hidden

        self.mu_w, self.sig_w, self.mu_b, self.sig_b = init_weights((self.D_hidden, self.D_out))
        
        self.l1 = torch.nn.Linear(D_in, D_hidden)
        self.l2 = torch.nn.Linear(D_hidden, D_hidden)
    
    def forward(self, x, is_train):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
    
        return noisy_l(x, self.mu_w, self.sig_w, self.mu_b, self.sig_b, is_train)
    
class Noisy_CNN(torch.nn.Module):
    def __init__(self, D_in, D_out, D_hidden=512):
        super(Noisy_CNN, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        
        self.conv1 = torch.nn.Conv2d(in_channels=self.D_in[0], out_channels=32, kernel_size=8, stride=4)
        dim1 = ((self.D_in[1] - 8)//4 + 1, (self.D_in[2] - 8)//4 + 1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        dim2 = ((dim1[0] - 4)//2 + 1, (dim1[1] - 4)//2 + 1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        dim3 = ((dim2[0] - 3)//1 + 1, (dim2[1] - 3)//1 + 1)
        self.fc = torch.nn.Linear(64*dim3[0]*dim3[1], D_hidden)
        
        self.mu_w1, self.sig_w1, self.mu_b1, self.sig_b1 = init_weights((D_hidden, D_hidden))
        self.mu_w2, self.sig_w2, self.mu_b2, self.sig_b2 = init_weights((D_hidden, self.D_out))
        
    def forward(self, x, is_train):
        x = (x-(255.0/2))/(255.0/2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        
        x = F.relu(noisy_l(x, self.mu_w1, self.sig_w1, self.mu_b1, self.sig_b1, is_train))
        x = noisy_l(x, self.mu_w2, self.sig_w2, self.mu_b2, self.sig_b2, is_train)
        return x