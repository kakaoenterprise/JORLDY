import torch
import torch.nn.functional as F

class RND(torch.nn.Module):
    def __init__(self, D_in, D_out, eta):
        super(RND, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.eta = eta
        
        feature_size = 256
        
        self.fc1_predict = torch.nn.Linear(self.D_in, 256)
        self.fc2_predict = torch.nn.Linear(256, feature_size)
        
        self.fc1_target = torch.nn.Linear(self.D_in, 256)
        self.fc2_target = torch.nn.Linear(256, feature_size)
                
        self.forward_loss = torch.nn.MSELoss()
            
    def forward(self, s_next):
        p = F.elu(self.fc1_predict(s_next))
        p = F.elu(self.fc2_predict(p))

        t = F.elu(self.fc1_target(s_next))
        t = F.elu(self.fc2_target(t))
        
        r_i = (self.eta * 0.5) * torch.sum(torch.square(p - t), axis = 1)
        l_f = self.forward_loss(p, t)
        
        return r_i, l_f
        
class RND_CNN(torch.nn.Module):
    def __init__(self, D_in, D_out, eta):
        super(RND_CNN, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.eta = eta
        
        self.conv1_predict = torch.nn.Conv2d(in_channels=self.D_in[0], out_channels=32, kernel_size=3, stride=2)
        self.conv2_predict = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.conv3_predict = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.conv4_predict = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        
        self.conv1_target = torch.nn.Conv2d(in_channels=self.D_in[0], out_channels=32, kernel_size=3, stride=2)
        self.conv2_target = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.conv3_target = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.conv4_target = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        
        dim1 = ((self.D_in[1] - 3)//2 + 1, (self.D_in[2] - 3)//2 + 1)
        dim2 = ((dim1[0] - 3)//2 + 1, (dim1[1] - 3)//2 + 1)
        dim3 = ((dim2[0] - 3)//2 + 1, (dim2[1] - 3)//2 + 1)
        dim4 = ((dim3[0] - 3)//2 + 1, (dim3[1] - 3)//2 + 1)
        
        feature_size = 32*dim4[0]*dim4[1]
        
        self.forward_loss = torch.nn.MSELoss()

    def forward(self, s_next):
        s_next = (s_next-(255.0/2))/(255.0/2)
        
        p = F.elu(self.conv1_predict(s_next))
        p = F.elu(self.conv2_predict(p))
        p = F.elu(self.conv3_predict(p))
        p = F.elu(self.conv4_predict(p))
        p = p.view(p.size(0), -1)
        
        t = F.elu(self.conv1_target(s_next))
        t = F.elu(self.conv2_target(t))
        t = F.elu(self.conv3_target(t))
        t = F.elu(self.conv4_target(t))
        t = t.view(t.size(0), -1)
        
        r_i = (self.eta * 0.5) * torch.sum(torch.square(p - t), axis = 1)
        
        l_f = self.forward_loss(p, t)

        return r_i, l_f
        