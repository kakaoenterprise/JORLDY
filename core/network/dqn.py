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
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.q(x)
    
class DQN_CNN(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(DQN_CNN, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
                
        self.conv1 = torch.nn.Conv2d(in_channels=self.D_in[0], out_channels=32, kernel_size=8, stride=4, padding=4)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(64*int(self.D_in[1]/8)*int(self.D_in[2]/8), 512)
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

class Dueling(torch.nn.Module):
    def __init__(self, D_in, D_out, D_hidden=512):
        super(Dueling, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        
        self.l1 = torch.nn.Linear(D_in, D_hidden)
        self.l2 = torch.nn.Linear(D_hidden, D_hidden)
        
        self.l1_a = torch.nn.Linear(D_hidden, D_hidden)
        self.l1_v = torch.nn.Linear(D_hidden, D_hidden)

        self.l2_a = torch.nn.Linear(D_hidden, self.D_out)
        self.l2_v = torch.nn.Linear(D_hidden, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        
        x_a = F.relu(self.l1_a(x))
        x_v = F.relu(self.l1_v(x))
        
        # A stream : action advantage
        x_a = self.l2_a(x_a) # [bs, num_action]
        x_a_mean = x_a.mean(dim=1).unsqueeze(1) # [bs, 1]
        x_a = x_a - x_a_mean.repeat(1, self.D_out) # [bs, num_action]

        # V stream : state value
        x_v = self.l2_v(x_v) # [bs, 1]
        x_v = x_v.repeat(1, self.D_out) # [bs, num_action]

        out = x_a + x_v # [bs, num_action]
        
        return out
    
class Dueling_CNN(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(Dueling_CNN, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        
        self.conv1 = torch.nn.Conv2d(in_channels=self.D_in[0], out_channels=32, kernel_size=8, stride=4, padding=4)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1)
        
        self.fc1_a = torch.nn.Linear(64*int(self.D_in[1]/8)*int(self.D_in[2]/8), 512)
        self.fc1_v = torch.nn.Linear(64*int(self.D_in[1]/8)*int(self.D_in[2]/8), 512)

        self.fc2_a = torch.nn.Linear(512, self.D_out)
        self.fc2_v = torch.nn.Linear(512, 1)
        
    def forward(self, x):
        x = (x-(255.0/2))/(255.0/2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        x_a = F.relu(self.fc1_a(x))
        x_v = F.relu(self.fc1_v(x))

        # A stream : action advantage
        x_a = self.fc2_a(x_a) # [bs, num_action]
        x_a_mean = x_a.mean(dim=1).unsqueeze(1) # [bs, 1]
        x_a = x_a - x_a_mean.repeat(1, self.num_action) # [bs, num_action]

        # V stream : state value
        x_v = self.fc2_v(x_v) # [bs, 1]
        x_v = x_v.repeat(1, self.num_action) # [bs, num_action]

        out = x_a + x_v # [bs, num_action]
        return out