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
        self.conv1 = torch.nn.Conv2d(in_channels=self.D_in[0], out_channels=32, kernel_size=8, stride=4)
        dim1 = ((self.D_in[1] - 8)//4 + 1, (self.D_in[2] - 8)//4 + 1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        dim2 = ((dim1[0] - 4)//2 + 1, (dim1[1] - 4)//2 + 1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        dim3 = ((dim2[0] - 3)//1 + 1, (dim2[1] - 3)//1 + 1)
        
        self.fc1 = torch.nn.Linear(64*dim3[0]*dim3[1], 512)
        self.fc2 = torch.nn.Linear(512, self.D_out)
        
    def forward(self, x):
        x = (x-(255.0/2))/(255.0/2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
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
        
        self.conv1 = torch.nn.Conv2d(in_channels=self.D_in[0], out_channels=32, kernel_size=8, stride=4)
        dim1 = ((self.D_in[1] - 8)//4 + 1, (self.D_in[2] - 8)//4 + 1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        dim2 = ((dim1[0] - 4)//2 + 1, (dim1[1] - 4)//2 + 1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        dim3 = ((dim2[0] - 3)//1 + 1, (dim2[1] - 3)//1 + 1)
        
        self.fc1_a = torch.nn.Linear(64*dim3[0]*dim3[1], 512)
        self.fc1_v = torch.nn.Linear(64*dim3[0]*dim3[1], 512)

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
    
class IQN(torch.nn.Module):
    def __init__(self, D_in, D_out, D_em, N_sample, D_hidden=512):
        super(IQN, self).__init__()
        self.D_in = D_in
        self.D_out = D_out * N_sample
        self.D_em = D_em 
        self.N_sample = N_sample
        
        self.state_embed = torch.nn.Linear(self.D_in, D_hidden)
        self.sample_embed = torch.nn.Linear(self.D_em, D_hidden)
        
        self.l1 = torch.nn.Linear(D_hidden, D_hidden)
        self.l2 = torch.nn.Linear(D_hidden, D_hidden)
        self.q = torch.nn.Linear(D_hidden, D_out)

    def forward(self, x, x_embed):               
        state_embed = F.relu(self.state_embed(x))
        state_embed_tile = state_embed.repeat(self.N_sample, 1)
        
        sample_embed_out = F.relu(self.sample_embed(x_embed))  

        embed_out = state_embed_tile * sample_embed_out
        
        x = F.relu(self.l1(embed_out))
        x = F.relu(self.l2(x))
        return self.q(x)
    
class IQN_CNN(torch.nn.Module):
    def __init__(self, D_in, D_out, D_em, N_sample):
        super(IQN_CNN, self).__init__()
        self.D_in = D_in
        self.D_out = D_out * N_sample
        self.D_em = D_em 
        self.N_sample = N_sample
        
        self.conv1 = torch.nn.Conv2d(in_channels=self.D_in[0], out_channels=32, kernel_size=8, stride=4)
        dim1 = ((self.D_in[1] - 8)//4 + 1, (self.D_in[2] - 8)//4 + 1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        dim2 = ((dim1[0] - 4)//2 + 1, (dim1[1] - 4)//2 + 1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        dim3 = ((dim2[0] - 3)//1 + 1, (dim2[1] - 3)//1 + 1)
        
        self.fc_embed = torch.nn.Linear(self.D_em, 64*dim3[0]*dim3[1])
        
        self.fc1 = torch.nn.Linear(64*dim3[0]*dim3[1], 512)
        self.fc2 = torch.nn.Linear(512, self.D_out)
        
    def forward(self, x, x_embed):
        x = (x-(255.0/2))/(255.0/2)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        h_flat = x.view(x.size(0), -1)
        h_flat_tile = x_flat.repeat(self.N_sample, 1)
        h_flat_embedding = h_flat_tile * x_embed
                           
        x = F.relu(self.fc1(h_flat_embedding))
        x = self.fc2(x)
        
        return x