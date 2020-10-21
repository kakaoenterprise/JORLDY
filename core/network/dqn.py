import torch
import torch.nn.functional as F

class DQN(torch.nn.Module):
    def __init__(self, D_in, D_out, D_hidden=512):
        super(DQN, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        
        self.l1 = torch.nn.Linear(D_in, D_hidden)
        self.l2 = torch.nn.Linear(D_hidden, D_hidden)
        self.q = torch.nn.Linear(D_hidden, D_out)

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.q(x)
    
class DQN_CNN(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(DQN_CNN, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        
        self.conv1 = torch.nn.Conv2d(in_channels=self.D_in[2], out_channels=32, kernel_size=8, stride=4, padding=4)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(64*int(self.D_in[0]/8)*int(self.D_in[1]/8), 512)
        self.fc2 = torch.nn.Linear(512, self.D_out)
        
    def forward(self, x):
        x = (x-(255.0/2))/(255.0/2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64*int(self.D_in[0]/8)*int(self.D_in[1]/8))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x