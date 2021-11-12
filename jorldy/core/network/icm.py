import torch
import torch.nn.functional as F

from .rnd import *

class ICM_MLP(torch.nn.Module):
    def __init__(self, D_in, D_out, num_workers, gamma, eta, action_type, 
                 ri_normalize=True, obs_normalize=True, batch_norm=True,
                 D_hidden=256):
        super(ICM_MLP, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.num_workers = num_workers
        self.eta = eta
        self.action_type = action_type
                
        self.rms_obs = RunningMeanStd(D_in)
        self.rms_ri = RunningMeanStd(1)
        self.rff = RewardForwardFilter(gamma, num_workers)
        self.obs_normalize = obs_normalize
        self.ri_normalize = ri_normalize
        self.batch_norm = batch_norm
        
        feature_size = 256
        
        self.fc1 = torch.nn.Linear(self.D_in, D_hidden)
        self.fc2 = torch.nn.Linear(D_hidden, feature_size)
        
        self.inverse_fc1 = torch.nn.Linear(2*feature_size, D_hidden)
        self.inverse_fc2 = torch.nn.Linear(D_hidden, self.D_out)
        
        self.forward_loss = torch.nn.MSELoss()
        
        if self.action_type == 'discrete':
            self.forward_fc1 = torch.nn.Linear(feature_size+1, D_hidden)
            self.forward_fc2 = torch.nn.Linear(D_hidden+1, feature_size)
        
            self.inverse_loss = torch.nn.CrossEntropyLoss()
        else:
            self.forward_fc1 = torch.nn.Linear(feature_size+self.D_out, D_hidden)
            self.forward_fc2 = torch.nn.Linear(D_hidden+self.D_out, feature_size)
        
            self.inverse_loss = torch.nn.MSELoss()
            
        if self.batch_norm:
            self.bn1 = torch.nn.BatchNorm1d(D_hidden)
            self.bn2 = torch.nn.BatchNorm1d(feature_size)
            
            self.bn1_next = torch.nn.BatchNorm1d(D_hidden)
     
    def update_rms_obs(self, v):
        self.rms_obs.update(v)
        
    def update_rms_ri(self, v):
        self.rms_ri.update(v)
        
    def forward(self, s, a, s_next, update_ri=False):
        if self.obs_normalize: s = normalize_obs(s, self.rms_obs.mean, self.rms_obs.var)
        if self.obs_normalize: s_next = normalize_obs(s_next, self.rms_obs.mean, self.rms_obs.var)
        
        if self.batch_norm:
            s = F.elu(self.bn1(self.fc1(s)))
            s = F.elu(self.bn2(self.fc2(s)))

            s_next = F.elu(self.bn1_next(self.fc1(s_next)))
        else:
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
        
        if update_ri:
            ri_T = r_i.view(self.num_workers, -1).T # (n_batch, n_workers)
            rewems = torch.stack([self.rff.update(rit.detach()) for rit in ri_T]).ravel() # (n_batch, n_workers) -> (n_batch * n_workers)
            self.update_rms_ri(rewems)
        if self.ri_normalize: r_i = r_i / (torch.sqrt(self.rms_ri.var) + 1e-7)
        
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
    def __init__(self, D_in, D_out, num_workers, gamma, eta, action_type,
                 ri_normalize=True, obs_normalize=True, batch_norm=True,
                 D_hidden=256):
        super(ICM_CNN, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.num_workers = num_workers
        self.eta = eta
        self.action_type = action_type

        self.rms_obs = RunningMeanStd(D_in)
        self.rms_ri = RunningMeanStd(1)
        self.rff = RewardForwardFilter(gamma, num_workers)
        self.obs_normalize = obs_normalize
        self.ri_normalize = ri_normalize
        self.batch_norm = batch_norm
        
        self.conv1 = torch.nn.Conv2d(in_channels=self.D_in[0], out_channels=32, kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)

        dim1 = ((self.D_in[1] - 3)//2 + 1, (self.D_in[2] - 3)//2 + 1)
        dim2 = ((dim1[0] - 3)//2 + 1, (dim1[1] - 3)//2 + 1)
        dim3 = ((dim2[0] - 3)//2 + 1, (dim2[1] - 3)//2 + 1)
        dim4 = ((dim3[0] - 3)//2 + 1, (dim3[1] - 3)//2 + 1)
        
        feature_size = 32*dim4[0]*dim4[1]
        
        self.inverse_fc1 = torch.nn.Linear(2*feature_size, D_hidden)
        self.inverse_fc2 = torch.nn.Linear(D_hidden, self.D_out)
        
        self.forward_loss = torch.nn.MSELoss()
        
        if self.action_type == 'discrete':
            self.forward_fc1 = torch.nn.Linear(feature_size+1, D_hidden)
            self.forward_fc2 = torch.nn.Linear(D_hidden+1, feature_size)
        
            self.inverse_loss = torch.nn.CrossEntropyLoss()
        else:
            self.forward_fc1 = torch.nn.Linear(feature_size+self.D_out, D_hidden)
            self.forward_fc2 = torch.nn.Linear(D_hidden+self.D_out, feature_size)
        
            self.inverse_loss = torch.nn.MSELoss()   
            
        if self.batch_norm:
            self.bn1 = torch.nn.BatchNorm2d(32)
            self.bn2 = torch.nn.BatchNorm2d(32)
            self.bn3 = torch.nn.BatchNorm2d(32)
            self.bn4 = torch.nn.BatchNorm2d(32)
                        
            self.bn1_next = torch.nn.BatchNorm2d(32)
            self.bn2_next = torch.nn.BatchNorm2d(32)
            self.bn3_next = torch.nn.BatchNorm2d(32)
            
    def update_rms_obs(self, v):
        self.rms_obs.update(v/255.0)
        
    def update_rms_ri(self, v):
        self.rms_ri.update(v)
        
    def forward(self, s, a, s_next, update_ri=False):
        if self.obs_normalize: s = normalize_obs(s, self.rms_obs.mean, self.rms_obs.var)
        if self.obs_normalize: s_next = normalize_obs(s_next, self.rms_obs.mean, self.rms_obs.var)
        
        if self.batch_norm:
            s = F.elu(self.bn1(self.conv1(s)))
            s = F.elu(self.bn2(self.conv2(s)))
            s = F.elu(self.bn3(self.conv3(s)))
            s = F.elu(self.bn4(self.conv4(s)))

            s_next = F.elu(self.bn1_next(self.conv1(s_next)))
            s_next = F.elu(self.bn2_next(self.conv2(s_next)))
            s_next = F.elu(self.bn3_next(self.conv3(s_next)))
            
        else:
            s = F.elu(self.conv1(s))
            s = F.elu(self.conv2(s))
            s = F.elu(self.conv3(s))
            s = F.elu(self.conv4(s))

            s_next = F.elu(self.conv1(s_next))
            s_next = F.elu(self.conv2(s_next))
            s_next = F.elu(self.conv3(s_next))
        
        s_next = F.elu(self.conv4(s_next))
        s = s.view(s.size(0), -1)
        s_next = s_next.view(s_next.size(0), -1)
        
        # Forward Model
        x_forward = torch.cat((s, a), axis=1)
        x_forward = F.relu(self.forward_fc1(x_forward))
        x_forward = torch.cat((x_forward, a), axis=1)
        x_forward = self.forward_fc2(x_forward)
        
        r_i = (self.eta * 0.5) * torch.sum(torch.abs(x_forward - s_next), axis = 1)
        
        if update_ri:
            ri_T = r_i.view(self.num_workers, -1).T # (n_batch, n_workers)
            rewems = torch.stack([self.rff.update(rit.detach()) for rit in ri_T]).ravel() # (n_batch, n_workers) -> (n_batch * n_workers)
            self.update_rms_ri(rewems)
        if self.ri_normalize: r_i = r_i / (torch.sqrt(self.rms_ri.var) + 1e-7)
        
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
        
class ICM_Multi(torch.nn.Module):
    def __init__(self, D_in, D_out, num_workers, gamma, eta, action_type,
                 ri_normalize=True, obs_normalize=True, batch_norm=True,
                 D_hidden=256):
        super(ICM_Multi, self).__init__()
        self.D_in_img = D_in[0]
        self.D_in_vec = D_in[1]
        
        self.D_out = D_out
        self.num_workers = num_workers
        self.eta = eta
        self.action_type = action_type

        self.rms_obs_img = RunningMeanStd(self.D_in_img)
        self.rms_obs_vec = RunningMeanStd(self.D_in_vec)
        
        self.rms_ri = RunningMeanStd(1)
        self.rff = RewardForwardFilter(gamma, num_workers)
        self.obs_normalize = obs_normalize
        self.ri_normalize = ri_normalize
        self.batch_norm = batch_norm
        
        ################################## Conv HEAD ##################################
        self.conv1 = torch.nn.Conv2d(in_channels=self.D_in_img[0], out_channels=32, kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)

        dim1 = ((self.D_in_img[1] - 3)//2 + 1, (self.D_in_img[2] - 3)//2 + 1)
        dim2 = ((dim1[0] - 3)//2 + 1, (dim1[1] - 3)//2 + 1)
        dim3 = ((dim2[0] - 3)//2 + 1, (dim2[1] - 3)//2 + 1)
        dim4 = ((dim3[0] - 3)//2 + 1, (dim3[1] - 3)//2 + 1)
        
        feature_size_img = 32*dim4[0]*dim4[1]
        
        ################################## MLP HEAD ##################################        
        feature_size_mlp = 256

        self.fc1_mlp = torch.nn.Linear(self.D_in_vec, D_hidden)
        self.fc2_mlp = torch.nn.Linear(D_hidden, feature_size_mlp)        
        ##############################################################################
        
        feature_size = feature_size_img + feature_size_mlp
        
        self.inverse_fc1 = torch.nn.Linear(2*feature_size, D_hidden)
        self.inverse_fc2 = torch.nn.Linear(D_hidden, self.D_out)
        
        self.forward_loss = torch.nn.MSELoss()
        
        if self.action_type == 'discrete':
            self.forward_fc1 = torch.nn.Linear(feature_size+1, D_hidden)
            self.forward_fc2 = torch.nn.Linear(D_hidden+1, feature_size)
        
            self.inverse_loss = torch.nn.CrossEntropyLoss()
        else:
            self.forward_fc1 = torch.nn.Linear(feature_size+self.D_out, D_hidden)
            self.forward_fc2 = torch.nn.Linear(D_hidden+self.D_out, feature_size)
        
            self.inverse_loss = torch.nn.MSELoss()   
            
        if self.batch_norm:
            self.bn1_conv = torch.nn.BatchNorm2d(32)
            self.bn2_conv = torch.nn.BatchNorm2d(32)
            self.bn3_conv = torch.nn.BatchNorm2d(32)
            self.bn4_conv = torch.nn.BatchNorm2d(32)
                        
            self.bn1_next_conv = torch.nn.BatchNorm2d(32)
            self.bn2_next_conv = torch.nn.BatchNorm2d(32)
            self.bn3_next_conv = torch.nn.BatchNorm2d(32)
            
            self.bn1_mlp = torch.nn.BatchNorm1d(D_hidden)
            self.bn2_mlp = torch.nn.BatchNorm1d(feature_size_mlp)
            
            self.bn1_next_mlp = torch.nn.BatchNorm1d(D_hidden)
            
    def update_rms_obs(self, v):
        self.rms_obs_img.update(v[0]/255.0)
        self.rms_obs_vec.update(v[1])
        
    def update_rms_ri(self, v):
        self.rms_ri.update(v)
        
    def forward(self, s, a, s_next, update_ri=False):
        s_img = s[0]
        s_vec = s[1]
        
        s_next_img = s_next[0]
        s_next_vec = s_next[1]

        if self.obs_normalize: 
            s_img = normalize_obs(s_img, self.rms_obs_img.mean, self.rms_obs_img.var)
            s_vec = normalize_obs(s_vec, self.rms_obs_vec.mean, self.rms_obs_vec.var)
            s_next_img = normalize_obs(s_next_img, self.rms_obs_img.mean, self.rms_obs_img.var)
            s_next_vec = normalize_obs(s_next_vec, self.rms_obs_vec.mean, self.rms_obs_vec.var)
            
        if self.batch_norm:
            s_img = F.elu(self.bn1_conv(self.conv1(s_img)))
            s_img = F.elu(self.bn2_conv(self.conv2(s_img)))
            s_img = F.elu(self.bn3_conv(self.conv3(s_img)))
            s_img = F.elu(self.bn4_conv(self.conv4(s_img)))

            s_next_img = F.elu(self.bn1_next_conv(self.conv1(s_next_img)))
            s_next_img = F.elu(self.bn2_next_conv(self.conv2(s_next_img)))
            s_next_img = F.elu(self.bn3_next_conv(self.conv3(s_next_img)))
            
            s_vec = F.elu(self.bn1_mlp(self.fc1_mlp(s_vec)))
            s_vec = F.elu(self.bn2_mlp(self.fc2_mlp(s_vec)))

            s_next_vec = F.elu(self.bn1_next_mlp(self.fc1_mlp(s_next_vec)))
        else:
            s_img = F.elu(self.conv1(s_img))
            s_img = F.elu(self.conv2(s_img))
            s_img = F.elu(self.conv3(s_img))
            s_img = F.elu(self.conv4(s_img))

            s_next_img = F.elu(self.conv1(s_next_img))
            s_next_img = F.elu(self.conv2(s_next_img))
            s_next_img = F.elu(self.conv3(s_next_img))
            
            s_vec = F.elu(self.fc1_mlp(s_vec))
            s_vec = F.elu(self.fc2_mlp(s_vec))

            s_next_vec = F.elu(self.fc1_mlp(s_next_vec))    
                           
        s_next_img = F.elu(self.conv4(s_next_img))
        s_img = s_img.view(s_img.size(0), -1)
        s_next_img = s_next_img.view(s_next_img.size(0), -1)
        
        s_next_vec = F.elu(self.fc2_mlp(s_next_vec))
        
        s = torch.cat((s_img, s_vec), -1)
        s_next = torch.cat((s_next_img, s_next_vec), -1)
        
        # Forward Model
        x_forward = torch.cat((s, a), axis=1)
        x_forward = F.relu(self.forward_fc1(x_forward))
        x_forward = torch.cat((x_forward, a), axis=1)
        x_forward = self.forward_fc2(x_forward)
        
        r_i = (self.eta * 0.5) * torch.sum(torch.abs(x_forward - s_next), axis = 1)
        
        if update_ri:
            ri_T = r_i.view(self.num_workers, -1).T # (n_batch, n_workers)
            rewems = torch.stack([self.rff.update(rit.detach()) for rit in ri_T]).ravel() # (n_batch, n_workers) -> (n_batch * n_workers)
            self.update_rms_ri(rewems)
        if self.ri_normalize: r_i = r_i / (torch.sqrt(self.rms_ri.var) + 1e-7)
        
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
        