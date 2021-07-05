import torch
import torch.nn.functional as F

class ICM(torch.nn.Module):
    def __init__(self, D_in, D_out, eta, action_type):
        super(ICM, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.eta = eta
        self.action_type = action_type
        
        feature_size = 256
        
        self.fc1 = torch.nn.Linear(self.D_in, 256)
        self.fc2 = torch.nn.Linear(256, feature_size)
        
        self.inverse_fc1 = torch.nn.Linear(2*feature_size, 256)
        self.inverse_fc2 = torch.nn.Linear(256, self.D_out)
        
        self.forward_loss = torch.nn.MSELoss()
        
        if self.action_type == 'discrete':
            self.forward_fc1 = torch.nn.Linear(feature_size+1, 256)
            self.forward_fc2 = torch.nn.Linear(256+1, feature_size)
        
            self.inverse_loss = torch.nn.CrossEntropyLoss()
        else:
            self.forward_fc1 = torch.nn.Linear(feature_size+self.D_out, 256)
            self.forward_fc2 = torch.nn.Linear(256+self.D_out, feature_size)
        
            self.inverse_loss = torch.nn.MSELoss()
            
    def forward(self, s, a, s_next):
        s = F.elu(self.fc1(s))
        s = F.elu(self.fc2(s))

        s_next = F.elu(self.fc1(s_next))
        s_next = F.elu(self.fc2(s_next))
        
        # Forward Model
        x_forward = torch.cat((s, a), axis=1)
        x_forward = F.relu(self.forward_fc1(x_forward))
        x_forward = torch.cat((x_forward, a), axis=1)
        x_forward = self.forward_fc2(x_forward)
        
        r_i = (self.eta * 0.5) * torch.sum(torch.abs(x_forward - s_next), axis = 1)
        
        l_f = self.forward_loss(x_forward, s_next.detach())
        
        # Inverse Model 
        x_inverse = torch.cat((s, s_next), axis=1)
        x_inverse = F.relu(self.inverse_fc1(x_inverse))
        
        if self.action_type == 'discrete':
            x_inverse = F.softmax(self.inverse_fc2(x_inverse), dim=1)
            l_i = self.inverse_loss(x_inverse, a.view(-1).long())
        else:
            x_inverse = self.inverse_fc2(x_inverse)
            l_i = self.inverse_loss(x_inverse, a)
        
        return r_i, l_f, l_i
        
class ICM_CNN(torch.nn.Module):
    def __init__(self, D_in, D_out, eta, action_type):
        super(ICM_CNN, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.eta = eta
        self.action_type = action_type
        
        self.conv1 = torch.nn.Conv2d(in_channels=self.D_in[0], out_channels=32, kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)

        dim1 = ((self.D_in[1] - 3)//2 + 1, (self.D_in[2] - 3)//2 + 1)
        dim2 = ((dim1[0] - 3)//2 + 1, (dim1[1] - 3)//2 + 1)
        dim3 = ((dim2[0] - 3)//2 + 1, (dim2[1] - 3)//2 + 1)
        dim4 = ((dim3[0] - 3)//2 + 1, (dim3[1] - 3)//2 + 1)
        
        feature_size = 32*dim4[0]*dim4[1]
        
        self.inverse_fc1 = torch.nn.Linear(2*feature_size, 256)
        self.inverse_fc2 = torch.nn.Linear(256, self.D_out)
        
        self.forward_loss = torch.nn.MSELoss()
        
        if self.action_type == 'discrete':
            self.forward_fc1 = torch.nn.Linear(feature_size+1, 256)
            self.forward_fc2 = torch.nn.Linear(256+1, feature_size)
        
            self.inverse_loss = torch.nn.CrossEntropyLoss()
        else:
            self.forward_fc1 = torch.nn.Linear(feature_size+self.D_out, 256)
            self.forward_fc2 = torch.nn.Linear(256+self.D_out, feature_size)
        
            self.inverse_loss = torch.nn.MSELoss()   
            
    def forward(self, s, a, s_next):
        s = (s-(255.0/2))/(255.0/2)
        s_next = (s_next-(255.0/2))/(255.0/2)
        
        s = F.elu(self.conv1(s))
        s = F.elu(self.conv2(s))
        s = F.elu(self.conv3(s))
        s = F.elu(self.conv4(s))
        s = s.view(s.size(0), -1)
        
        s_next = F.elu(self.conv1(s_next))
        s_next = F.elu(self.conv2(s_next))
        s_next = F.elu(self.conv3(s_next))
        s_next = F.elu(self.conv4(s_next))
        s_next = s_next.view(s_next.size(0), -1)
        
        # Forward Model
        x_forward = torch.cat((s, a), axis=1)
        x_forward = F.relu(self.forward_fc1(x_forward))
        x_forward = torch.cat((x_forward, a), axis=1)
        x_forward = self.forward_fc2(x_forward)
        
        r_i = (self.eta * 0.5) * torch.sum(torch.abs(x_forward - s_next), axis = 1)
        
        l_f = self.forward_loss(x_forward, s_next.detach())
        
        # Inverse Model 
        x_inverse = torch.cat((s, s_next), axis=1)
        x_inverse = F.relu(self.inverse_fc1(x_inverse))
        
        if self.action_type == 'discrete':
            x_inverse = F.softmax(self.inverse_fc2(x_inverse), dim=1)
            l_i = self.inverse_loss(x_inverse, a.view(-1).long())
        else:
            x_inverse = self.inverse_fc2(x_inverse)
            l_i = self.inverse_loss(x_inverse, a)
        
        return r_i, l_f, l_i
        