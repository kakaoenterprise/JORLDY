import torch
import torch.nn.functional as F

# codes from https://github.com/openai/random-network-distillation
class RewardForwardFilter(torch.nn.Module):
    def __init__(self, gamma, num_workers):
        super(RewardForwardFilter, self).__init__()
        self.rewems = torch.nn.Parameter(torch.zeros(num_workers), requires_grad=False)
        self.gamma = gamma
        
    def update(self, rews):
        self.rewems.data = self.rewems * self.gamma + rews
        return self.rewems
    
# codes modified from https://github.com/openai/random-network-distillation
class RunningMeanStd(torch.nn.Module):
    def __init__(self, shape, epsilon=1e-4):
        super(RunningMeanStd, self).__init__()
                
        self.mean = torch.nn.Parameter(torch.zeros(shape), requires_grad=False)
        self.var = torch.nn.Parameter(torch.zeros(shape), requires_grad=False)
        self.count = torch.nn.Parameter(torch.tensor(epsilon), requires_grad=False)
        
    def update(self, x):
        batch_mean, batch_std, batch_count = x.mean(axis=0), x.std(axis=0), x.shape[0]
        batch_var = torch.square(batch_std)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean.data = new_mean
        self.var.data = new_var
        self.count.data = new_count
    