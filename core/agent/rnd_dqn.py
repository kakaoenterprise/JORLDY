import torch
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
import numpy as np
import copy

from core.network import Network
from core.optimizer import Optimizer
from .utils import ReplayBuffer
from .dqn import DQNAgent

class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma
        
    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems
    
class RunningMeanStd(object):
    def __init__(self, device="", epsilon=1e-4):
        self.mean = None
        self.var = None
        self.device=device
        self.count = epsilon

    def update(self, x):
        shape = x.shape[1:]
        if self.mean == None or self.var == None:
            self.mean = torch.zeros(shape, device=self.device)
            self.var = torch.zeros(shape, device=self.device)
        
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

class RNDDQNAgent(DQNAgent):
    def __init__(self,
                state_size,
                action_size,
                network='dqn',
                optimizer='adam',
                learning_rate=3e-4,
#                 opt_eps=1e-8,
#                 gamma=0.99,
#                 explore_step=90000,
#                 buffer_size=50000,
#                 batch_size=64,
#                 start_train_step=2000,
#                 target_update_period=500,
                device=None,
                # Parameters for Random Network Distillation
                rnd_network="rnd_cnn",
                gamma_i=0.99,
                extrinsic_coeff=1.0,
                intrinsic_coeff=1.0,
                **kwargs,
                ):
        super(RNDDQNAgent, self).__init__(state_size=state_size,
                                          action_size=action_size,
                                          network=network,
                                          **kwargs)
        
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gamma_i = gamma_i
        self.extrinsic_coeff = extrinsic_coeff
        self.intrinsic_coeff = intrinsic_coeff
        
        self.rff = RewardForwardFilter(self.gamma_i)
        self.rff_rms = RunningMeanStd(self.device)
        self.obs_rms = RunningMeanStd(self.device)
        
        self.rnd = Network(rnd_network, state_size, action_size, self.obs_rms).to(self.device)
        self.rnd_optimizer = Optimizer('adam', self.rnd.parameters(), lr=learning_rate)
        
        # Freeze random network
        for name, param in self.rnd.named_parameters():
            if "target" in name:
                param.requires_grad = False
                
    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training)
        
        if training and self.memory.size < max(self.batch_size, self.start_train_step):
            action = np.random.randint(0, self.action_size, size=(state.shape[0], 1))
        else:
            action = torch.argmax(self.network(torch.as_tensor(state, dtype=torch.float32, device=self.device)), -1, keepdim=True).cpu().numpy()
        return action

    def learn(self):
        transitions = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = map(lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device), transitions)
        
        reward *= self.extrinsic_coeff
        
        # RND: calculate exploration reward, update moments of obs and r_i
        self.rnd.update_rms(next_state.detach())
        r_i = self.rnd.forward(next_state)
        rewems = self.rff.update(r_i.detach())
        self.rff_rms.update(rewems)
        r_i = r_i / (torch.sqrt(self.rff_rms.var) + 1e-7)
        r_i = r_i.unsqueeze(-1)
        
        r_i *= self.intrinsic_coeff
        
        eye = torch.eye(self.action_size).to(self.device)
        one_hot_action = eye[action.view(-1).long()]
        qe = (self.network(state) * one_hot_action).sum(1, keepdims=True)
        qi = (self.network.get_qi(state) * one_hot_action).sum(1, keepdims=True)

        with torch.no_grad():
            max_Q = torch.max(qe).item()
            next_qe = self.target_network(next_state)
            next_qi = self.target_network.get_qi(next_state)
            
            target_qe = reward + (1 - done) * self.gamma * next_qe.max(1, keepdims=True).values
            target_qi = r_i + (1 - done) * self.gamma_i * next_qi.max(1, keepdims=True).values
            
        loss = F.smooth_l1_loss(qe, target_qe) + F.smooth_l1_loss(qi, target_qi)
        
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        
        loss_i = r_i.mean()
        
        self.rnd_optimizer.zero_grad(set_to_none=True)
        loss_i.backward()
        self.rnd_optimizer.step()
        
        self.num_learn += 1

        result = {
            "loss" : loss.item(),
            "max_Q": max_Q,
            "r_i": r_i.mean().item()
        }

        return result
    
    def process(self, transitions, step):
        result = {}

        # Process per step
        self.memory.store(transitions)
        delta_t = step - self.time_t
        self.time_t = step
        self.target_update_stamp += delta_t

        if self.memory.size > self.batch_size and self.time_t >= self.start_train_step:
            result = self.learn()

        # Process per step if train start
        if self.num_learn > 0:
            if self.target_update_stamp > self.target_update_period:
                self.update_target()
                self.target_update_stamp = 0

        return result