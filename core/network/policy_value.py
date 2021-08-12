import torch
import torch.nn.functional as F

class ContinuousPolicyValue(torch.nn.Module):
    def __init__(self, D_in, D_out, D_hidden=512):
        super(ContinuousPolicyValue, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        
        self.l1 = torch.nn.Linear(D_in, D_hidden)
        self.l2 = torch.nn.Linear(D_hidden, D_hidden)
        self.mu = torch.nn.Linear(D_hidden, D_out)
        self.log_std = torch.nn.Linear(D_hidden, D_out)
        self.v = torch.nn.Linear(D_hidden, 1)
        self.v_i = torch.nn.Linear(D_hidden, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        
        mu = torch.clamp(self.mu(x), min=-5., max=5.)
        log_std = torch.tanh(self.log_std(x))
        return mu, log_std.exp(), self.v(x)
    
    def get_vi(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.v_i(x)
    
    
class DiscretePolicyValue(torch.nn.Module):
    def __init__(self, D_in, D_out, D_hidden=512):
        super(DiscretePolicyValue, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        
        self.l1 = torch.nn.Linear(D_in, D_hidden)
        self.l2 = torch.nn.Linear(D_hidden, D_hidden)
        self.pi = torch.nn.Linear(D_hidden, D_out)
        self.v = torch.nn.Linear(D_hidden, 1)
        self.v_i = torch.nn.Linear(D_hidden, 1)
        
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return F.softmax(self.pi(x), dim=-1), self.v(x)

    def get_vi(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.v_i(x)
    
    
class ContinuousPolicyValue_CNN(torch.nn.Module):
    def __init__(self, D_in, D_out, D_hidden=512):
        super(ContinuousPolicyValue_CNN, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        
        self.conv1 = torch.nn.Conv2d(in_channels=D_in[0], out_channels=32, kernel_size=8, stride=4)
        dim1 = ((D_in[1] - 8)//4 + 1, (D_in[2] - 8)//4 + 1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        dim2 = ((dim1[0] - 4)//2 + 1, (dim1[1] - 4)//2 + 1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        dim3 = ((dim2[0] - 3)//1 + 1, (dim2[1] - 3)//1 + 1)
        
        self.fc1 = torch.nn.Linear(64*dim3[0]*dim3[1], D_hidden)
        self.mu = torch.nn.Linear(D_hidden, D_out)
        self.log_std = torch.nn.Linear(D_hidden, D_out)
        self.v = torch.nn.Linear(D_hidden, 1)
        self.v_i = torch.nn.Linear(D_hidden, 1)

    def forward(self, x):
        x = (x-(255.0/2))/(255.0/2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        
        mu = torch.clamp(self.mu(x), min=-5., max=5.)
        log_std = torch.tanh(self.log_std(x))
        return mu, log_std.exp(), self.v(x)
    
    def get_vi(self, x):
        x = (x-(255.0/2))/(255.0/2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        return self.v_i(x)
    
    
class DiscretePolicyValue_CNN(torch.nn.Module):
    def __init__(self, D_in, D_out, D_hidden=512):
        super(DiscretePolicyValue_CNN, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        
        self.conv1 = torch.nn.Conv2d(in_channels=D_in[0], out_channels=32, kernel_size=8, stride=4)
        dim1 = ((D_in[1] - 8)//4 + 1, (D_in[2] - 8)//4 + 1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        dim2 = ((dim1[0] - 4)//2 + 1, (dim1[1] - 4)//2 + 1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        dim3 = ((dim2[0] - 3)//1 + 1, (dim2[1] - 3)//1 + 1)
        
        self.fc1 = torch.nn.Linear(64*dim3[0]*dim3[1], D_hidden)
        
        self.pi = torch.nn.Linear(D_hidden, D_out)
        self.v = torch.nn.Linear(D_hidden, 1)
        self.v_i = torch.nn.Linear(D_hidden, 1)
        
    def forward(self, x):
        x = (x-(255.0/2))/(255.0/2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        
        return F.softmax(self.pi(x), dim=-1), self.v(x)
    
    def get_vi(self, x):
        x = (x-(255.0/2))/(255.0/2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        
        return self.v_i(x)