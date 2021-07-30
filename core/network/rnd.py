import torch
import torch.nn.functional as F

# codes from https://github.com/openai/random-network-distillation
class RewardForwardFilter(object):
    def __init__(self, n_step, device, gamma):
        self.rewems = torch.zeros(1, device=device, requires_grad=False)
        self.n_step = n_step
        self.gamma = gamma
        
    def update(self, rews):
        if self.rewems.shape[0] < rews.shape[0]:
            assert rews.shape[0] % self.rewems.shape[0] == 0
            self.rewems = self.rewems.repeat(rews.shape[0] // self.rewems.shape[0])
        
        self.rewems = self.rewems * self.gamma + rews
        return self.rewems
    
    def save(self):
        return {
            "rewems": self.rewems.cpu().numpy(),
            "gamma": self.gamma,
        }
    
    def load(self, v, device="gpu"):
        self.rewems = torch.tensor(v["rewems"], requires_grad=False, device=device)
        self.gamma = v["gamma"]
    
# codes modified from https://github.com/openai/random-network-distillation
class RunningMeanStd(object):
    def __init__(self, shape, device="gpu", epsilon=1e-4):
#         self.mean = None
#         self.var = None
        self.mean = torch.zeros(shape, device=device, requires_grad=False)
        self.var = torch.zeros(shape, device=device, requires_grad=False)
        
        self.count = epsilon

    def update(self, x):
#         shape = x.shape[1:]
#         if self.mean == None or self.var == None:
#             self.mean = torch.zeros(shape, device=self.device, requires_grad=False)
#             self.var = torch.zeros(shape, device=self.device, requires_grad=False)
        
        batch_mean, batch_std, batch_count = x.mean(axis=0), x.std(axis=0), x.shape[0]
        batch_var = torch.square(batch_std)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
        
    def save(self):
        return {
            "mean": self.mean.cpu().numpy(),
            "var": self.var.cpu().numpy(),
            "count": self.count,
        }
    
    def load(self, v, device="gpu"):
        self.mean = torch.tensor(v["mean"], requires_grad=False, device=device)
        self.var = torch.tensor(v["var"], requires_grad=False, device=device)
        self.count = v["count"]

# normalize observation
# assumed state shape: (batch_size, dim_state)
def normalize_obs(obs, m, v):    
    return torch.clip((obs - m) / (torch.sqrt(v)+1e-7), min=-5., max=5.)

class RND(torch.nn.Module):
    def __init__(self, D_in, D_out, n_step, device_rms, gamma_i, 
                 ri_normalize=True, obs_normalize=True, batch_norm=True):
        super(RND, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.rms = {
            'obs': RunningMeanStd(D_in, device_rms),
            'ri': RunningMeanStd(1, device_rms),
        }
        self.n_step = n_step
        self.rff = RewardForwardFilter(n_step, device_rms, gamma_i)
        self.obs_normalize = obs_normalize
        self.ri_normalize = ri_normalize
        self.batch_norm = batch_norm
        
        feature_size = 256
        
        self.fc1_predict = torch.nn.Linear(self.D_in, 256)
        self.fc2_predict = torch.nn.Linear(256, feature_size)
        
        self.bn1_predict = torch.nn.BatchNorm1d(256)
        self.bn2_predict = torch.nn.BatchNorm1d(feature_size)
        
        self.fc1_target = torch.nn.Linear(self.D_in, 256)
        self.fc2_target = torch.nn.Linear(256, feature_size)
        
        self.bn1_target = torch.nn.BatchNorm1d(256)
        self.bn2_target = torch.nn.BatchNorm1d(feature_size)
        
    def update_rms(self, v, k='obs'):
        self.rms[k].update(v)
                            
    def forward(self, s_next, update_ri=False):
        if self.obs_normalize: s_next = normalize_obs(s_next, self.rms['obs'].mean, self.rms['obs'].var)
        p = F.elu(self.fc1_predict(s_next))
        if self.batch_norm: p = bn1_predict(p)
        p = F.elu(self.fc2_predict(p))
        if self.batch_norm: p = bn2_predict(p)

        t = F.elu(self.fc1_target(s_next))
        if self.batch_norm: t = bn1_predict(t)
        t = F.elu(self.fc2_target(t))
        if self.batch_norm: t = bn1_predict(t)
        
        r_i = torch.mean(torch.square(p - t), axis = 1)
        
        if update_ri:
            rewems = self.rff.update(r_i.detach())
            self.update_rms(rewems, 'ri')
        if self.ri_normalize: r_i = r_i / (torch.sqrt(self.rms['ri'].var) + 1e-7)
        
        return r_i
        
class RND_CNN(torch.nn.Module):
    def __init__(self, D_in, D_out, n_step, device_rms, gamma_i, 
                 ri_normalize=True, obs_normalize=True, batch_norm=True):
        super(RND_CNN, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.rms = {
            'obs': RunningMeanStd(D_in, device_rms),
            'ri': RunningMeanStd(1, device_rms),
        }
        self.n_step = n_step
        self.rff = RewardForwardFilter(n_step, device_rms, gamma_i)
        self.obs_normalize = obs_normalize
        self.ri_normalize = ri_normalize
        self.batch_norm = batch_norm
        
        # Predictor Networks
        self.conv1_predict = torch.nn.Conv2d(in_channels=self.D_in[0], out_channels=32, kernel_size=8, stride=4)
        self.conv2_predict = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3_predict = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        self.bn1_predict = torch.nn.BatchNorm2d(32)
        self.bn2_predict = torch.nn.BatchNorm2d(64)
        self.bn3_predict = torch.nn.BatchNorm2d(64)
        
        # Target Networks
        self.conv1_target = torch.nn.Conv2d(in_channels=self.D_in[0], out_channels=32, kernel_size=8, stride=4)
        self.conv2_target = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3_target = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.bn1_target = torch.nn.BatchNorm2d(32)
        self.bn2_target = torch.nn.BatchNorm2d(64)
        self.bn3_target = torch.nn.BatchNorm2d(64)
        
        dim1 = ((self.D_in[1] - 8)//4 + 1, (self.D_in[2] - 8)//4 + 1)
        dim2 = ((dim1[0] - 4)//2 + 1, (dim1[1] - 4)//2 + 1)
        dim3 = ((dim2[0] - 3)//1 + 1, (dim2[1] - 3)//1 + 1)
        
        feature_size = 64*dim3[0]*dim3[1]
        
    def update_rms(self, v, k='obs'):
        if k=='obs': v = v/255.0
        self.rms[k].update(v)
        
    def forward(self, s_next, update_ri=False):
        s_next = s_next/255.0
        if self.obs_normalize: s_next = normalize_obs(s_next, self.rms['obs'].mean, self.rms['obs'].var)
        
        p = F.relu(self.conv1_predict(s_next))
        if self.batch_norm: p = self.bn1_predict(p)
        p = F.relu(self.conv2_predict(p))
        if self.batch_norm: p = self.bn2_predict(p)
        p = F.relu(self.conv3_predict(p))
        if self.batch_norm: p = self.bn3_predict(p)
        p = p.view(p.size(0), -1)
        
        t = F.relu(self.conv1_target(s_next))
        if self.batch_norm: t = self.bn1_target(t)
        t = F.relu(self.conv2_target(t))
        if self.batch_norm: t = self.bn2_target(t)
        t = F.relu(self.conv3_target(t))
        if self.batch_norm: t = self.bn3_target(t)
        t = t.view(t.size(0), -1)
        
        r_i = torch.mean(torch.square(p - t), axis = 1)
        ri_T = r_i.view(-1, self.n_step).T # (n_step, n_workers)
        
        if update_ri:
            rewems = torch.stack([self.rff.update(rit.detach()) for rit in ri_T]).ravel() # (n_step, n_workers) -> (n_step * n_workers)
            self.update_rms(rewems, 'ri')
        if self.ri_normalize: r_i = r_i / (torch.sqrt(self.rms['ri'].var) + 1e-7)
        
        return r_i
        
class RND_RNN(torch.nn.Module):
    def __init__(self, D_in, D_out, n_step, device_rms, gamma_i, 
                 ri_normalize=True, obs_normalize=True, batch_norm=True):
        super(RND_RNN, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.rms = {
            'obs': RunningMeanStd(D_in, device_rms),
            'ri': RunningMeanStd(1, device_rms),
        }
        self.n_step = n_step
        self.rff = RewardForwardFilter(n_step, device_rms, gamma_i)
        self.obs_normalize = obs_normalize
        self.ri_normalize = ri_normalize
        
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
        
    def update_rms(self, v, k='obs'):
        if k=='obs': v = v/255.0
        self.rms[k].update(v)
        
    def forward(self, s_next, update_ri=False):
        s_next = s_next/255.0
        if self.obs_normalize: s_next = normalize_obs(s_next, self.rms['obs'].mean, self.rms['obs'].var)
        
        p = F.relu(self.conv1_predict(s_next))
        p = F.relu(self.conv2_predict(p))
        p = F.relu(self.conv3_predict(p))
        p = p.view(p.size(0), -1)
        
        t = F.relu(self.conv1_target(s_next))
        t = F.relu(self.conv2_target(t))
        t = F.relu(self.conv3_target(t))
        t = t.view(t.size(0), -1)
        
        r_i = torch.mean(torch.square(p - t), axis = 1)
        
        if update_ri:
            rewems = self.rff.update(r_i.detach())
            self.update_rms(rewems, 'ri')
            
        if self.ri_normalize: r_i = r_i / (torch.sqrt(self.rms['ri'].var) + 1e-7)
        
        return r_i