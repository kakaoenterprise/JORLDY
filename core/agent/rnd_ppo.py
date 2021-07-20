import torch
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np

from .reinforce import REINFORCEAgent
from core.optimizer import Optimizer
from core.network import Network

# RewardForwardFilter
# codes from https://github.com/openai/random-network-distillation
# line 543~552 of https://github.com/openai/random-network-distillation/blob/master/ppo_agent.py
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
    
# RunningMeanStd
# codes from https://github.com/openai/random-network-distillation
# modified from line 179~214 of https://github.com/openai/random-network-distillation/blob/master/mpi_util.py
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

class RNDPPOAgent(REINFORCEAgent):
    def __init__(self,
                 state_size,
                 action_size,
                 network="discrete_pi_v",
                 batch_size=32,
                 n_step=100,
                 n_epoch=5,
                 _lambda=0.9,
                 epsilon_clip=0.2,
                 vf_coef=0.5,
                 ent_coef=0.0,
                 # Parameters for Random Network Distillation
                 rnd_network="rnd_cnn",
                 gamma_i=0.99,
                 extrinsic_coeff=1.0,
                 intrinsic_coeff=1.0,
                 **kwargs,
                 ):
        super(RNDPPOAgent, self).__init__(state_size=state_size,
                                          action_size=action_size,
                                          network=network,
                                          **kwargs)
        self.batch_size = batch_size
        self.n_step = n_step
        self.n_epoch = n_epoch
        self._lambda = _lambda
        self.epsilon_clip = epsilon_clip
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.time_t = 0
        self.learn_stamp = 0
        
        self.gamma_i = gamma_i
        self.extrinsic_coeff = extrinsic_coeff
        self.intrinsic_coeff = intrinsic_coeff
        
        self.rff = RewardForwardFilter(self.gamma_i)
        self.rff_rms = RunningMeanStd(self.device)
        self.obs_rms = RunningMeanStd(self.device)
        
        self.rnd = Network(rnd_network, state_size, action_size, self.obs_rms).to(self.device)
        self.rnd_optimizer = Optimizer('adam', self.rnd.parameters(), lr=self.learning_rate)
        
        # Freeze random network
        for name, param in self.rnd.named_parameters():
            if "target" in name:
                param.requires_grad = False
                
    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training)
        
        if self.action_type == "continuous":
            mu, std, _ = self.network(torch.as_tensor(state, dtype=torch.float32, device=self.device))
            z = torch.normal(mu, std) if training else mu
            action = torch.tanh(z)
        else:
            pi, _ = self.network(torch.as_tensor(state, dtype=torch.float32, device=self.device))
            action = torch.multinomial(pi, 1) if training else torch.argmax(pi, dim=-1, keepdim=True)
        return action.cpu().numpy()

    def learn(self):
        transitions = self.memory.rollout()
        state, action, reward, next_state, done = map(lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device), transitions)
        
        # RND: calculate exploration reward, update moments of obs and r_i
        self.rnd.update_rms(next_state.detach())
        r_i = self.rnd.forward(next_state)
        rewems = self.rff.update(r_i.detach())
        self.rff_rms.update(rewems)
        r_i = r_i / (torch.sqrt(self.rff_rms.var) + 1e-7)
        
        # set pi_old and advantage
        with torch.no_grad():            
            if self.action_type == "continuous":
                mu, std, value = self.network(state)
                m = Normal(mu, std)
                z = torch.atanh(torch.clamp(action, -1+1e-7, 1-1e-7))
                pi = m.log_prob(z).exp()
            else:
                pi, value = self.network(state)
                pi = pi.gather(1, action.long())
            pi_old = pi
            v_i = self.network.get_vi(state)
            
            next_value = self.network(next_state)[-1]
            next_vi = self.network.get_vi(next_state)
            delta = reward + (1 - done) * self.gamma * next_value - value
            # non-episodic intrinsic reward, hence (1-done) not applied
            delta_i = r_i + self.gamma_i * next_vi - v_i
            adv = delta.clone() 
            adv_i = delta_i.clone()
            for t in reversed(range(len(adv))):
                if t > 0 and (t + 1) % self.n_step == 0:
                    continue
                adv[t] += (1 - done[t]) * self.gamma * self._lambda * adv[t+1]
                adv_i[t] += self.gamma_i * self._lambda * adv_i[t+1]
            adv = (adv - adv.mean()) / (adv.std() + 1e-7)
            adv_i = (adv_i - adv_i.mean()) / (adv_i.std() + 1e-7)
            ret = adv + value
            ret_i = adv_i + v_i
        
        # start train iteration
        actor_losses, critic_losses, entropy_losses, ratios, pis, pi_olds = [], [], [], [], [], []
        rnd_losses = []
        pi_olds.append(pi_old.min().item())
        idxs = np.arange(len(reward))
        for idx_epoch in range(self.n_epoch):
            np.random.shuffle(idxs)
            
            for offset in range(0, len(reward), self.batch_size):
                idx = idxs[offset : offset + self.batch_size]
                
                _state, _action, _ret, _next_state, _done, _adv, _pi_old =\
                    map(lambda x: x[idx], [state, action, ret, next_state, done, adv, pi_old])
                _ret_i, _adv_i = map(lambda x: x[idx], [ret_i, adv_i])

                _r_i = self.rnd.forward(_next_state)
                _r_i = _r_i / (_r_i.detach().std() + 1e-7)
                
                if self.action_type == "continuous":
                    mu, std, value = self.network(_state)
                    m = Normal(mu, std)
                    z = torch.atanh(torch.clamp(_action, -1+1e-7, 1-1e-7))
                    pi = m.log_prob(z).exp()
                else:
                    pi, value = self.network(_state)
                    pi = pi.gather(1, _action.long())
                _v_i = self.network.get_vi(_state)
                
                ratio = (pi / (_pi_old + 1e-4)).prod(1, keepdim=True)
                surr1 = ratio * (_adv + self.intrinsic_coeff * _adv_i)
                surr2 = torch.clamp(ratio, min=1-self.epsilon_clip, max=1+self.epsilon_clip) * (_adv + self.intrinsic_coeff * _adv_i)
                actor_loss = -torch.min(surr1, surr2).mean() 
                
                critic_loss = F.mse_loss(value, _ret).mean() + F.mse_loss(_v_i, _ret_i).mean()
                
                entropy_loss = torch.log(pi + 1e-4).mean()
                loss = actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss
                
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
                self.optimizer.step()
                
                rnd_loss = _r_i.mean()
                self.rnd_optimizer.zero_grad(set_to_none=True)
                rnd_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.rnd.parameters(), 1)
                self.rnd_optimizer.step()
                
                pis.append(pi.min().item())
                ratios.append(ratio.max().item())
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy_loss.item())
                rnd_losses.append(rnd_loss.item())
            
        result = {
            'actor_loss' : np.mean(actor_losses),
            'critic_loss' : np.mean(critic_losses),
            'entropy_loss' : np.mean(entropy_losses),
            'rnd_loss': np.mean(rnd_losses),
            'max_ratio' : max(ratios),
            'min_pi': min(pis),
            'min_pi_old': min(pi_olds),
        }
        return result

    def process(self, transitions, step):
        result = {}
        # Process per step
        self.memory.store(transitions)
        delta_t = step - self.time_t
        self.time_t = step
        self.learn_stamp += delta_t
        
        # Process per epi
        if self.learn_stamp >= self.n_step :
            result = self.learn()
            self.learn_stamp = 0
        
        return result
    

        
