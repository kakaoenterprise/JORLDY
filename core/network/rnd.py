import torch
import torch.nn.functional as F

# normalize observation
# assumed state shape: (batch_size, dim_state)
def normalize_obs(obs):
    m = obs.mean()
    s = obs.std()

    return torch.clip((obs - m) / (s+1e-7), min=-5., max=5.)

class RND(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(RND, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        
        feature_size = 256
        
        self.fc1_predict = torch.nn.Linear(self.D_in, 256)
        self.fc2_predict = torch.nn.Linear(256, feature_size)
        
        self.fc1_target = torch.nn.Linear(self.D_in, 256)
        self.fc2_target = torch.nn.Linear(256, feature_size)
                            
    def forward(self, s_next):
        s_next = normalize_obs(s_next)
        p = F.elu(self.fc1_predict(s_next))
        p = F.elu(self.fc2_predict(p))

        t = F.elu(self.fc1_target(s_next))
        t = F.elu(self.fc2_target(t))
        
        r_i = torch.mean(torch.square(p - t), axis = 1)
        
        return r_i
        
class RND_CNN(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(RND_CNN, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        
        # Predictor Networks
        self.conv1_predict = torch.nn.Conv2d(in_channels=self.D_in[0], out_channels=32, kernel_size=8, stride=4)
        self.conv2_predict = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3_predict = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # Target Networks
        self.conv1_target = torch.nn.Conv2d(in_channels=self.D_in[0], out_channels=32, kernel_size=8, stride=4)
        self.conv2_target = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3_target = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        dim1 = ((self.D_in[1] - 8)//4 + 1, (self.D_in[2] - 8)//4 + 1)
        dim2 = ((dim1[0] - 4)//2 + 1, (dim1[1] - 4)//2 + 1)
        dim3 = ((dim2[0] - 3)//1 + 1, (dim2[1] - 3)//1 + 1)
        
        feature_size = 64*dim3[0]*dim3[1]
        
    def forward(self, s_next):
        s_next = s_next/255.0
        s_next = normalize_obs(s_next)
        
        p = F.relu(self.conv1_predict(s_next))
        p = F.relu(self.conv2_predict(p))
        p = F.relu(self.conv3_predict(p))
        p = p.view(p.size(0), -1)
        
        t = F.relu(self.conv1_target(s_next))
        t = F.relu(self.conv2_target(t))
        t = F.relu(self.conv3_target(t))
        t = t.view(t.size(0), -1)
        
        r_i = torch.mean(torch.square(p - t), axis = 1)
        
        return r_i
        
class RND_RNN(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(RND_RNN, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        
        # Predictor Networks
        self.conv1_predict = torch.nn.Conv2d(in_channels=self.D_in[0], out_channels=32, kernel_size=8, stride=4)
        self.conv2_predict = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3_predict = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # Target Networks
        self.conv1_target = torch.nn.Conv2d(in_channels=self.D_in[0], out_channels=32, kernel_size=8, stride=4)
        self.conv2_target = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3_target = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        dim1 = ((self.D_in[1] - 8)//4 + 1, (self.D_in[2] - 8)//4 + 1)
        dim2 = ((dim1[0] - 4)//2 + 1, (dim1[1] - 4)//2 + 1)
        dim3 = ((dim2[0] - 3)//1 + 1, (dim2[1] - 3)//1 + 1)
        
        feature_size = 64*dim3[0]*dim3[1]
        
    def forward(self, s_next):
        s_next = s_next/255.0
        s_next = normalize_obs(s_next)
        
        p = F.relu(self.conv1_predict(s_next))
        p = F.relu(self.conv2_predict(p))
        p = F.relu(self.conv3_predict(p))
        p = p.view(p.size(0), -1)
        
        t = F.relu(self.conv1_target(s_next))
        t = F.relu(self.conv2_target(t))
        t = F.relu(self.conv3_target(t))
        t = t.view(t.size(0), -1)
        
        r_i = torch.mean(torch.square(p - t), axis = 1)
        
        return r_i