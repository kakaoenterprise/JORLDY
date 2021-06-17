import torch
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np

from .ppo import PPOAgent
from core.network import Network

class ICMPPOAgent(PPOAgent):
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
                 # Parameters for Curiosity-driven Exploration
                 icm_network= "icm",
                 beta = 0.2,
                 lamb = 1.0,
                 eta = 0.01,
                 extrinsic_coeff = 1.0,
                 intrinsic_coeff = 0.01,
                 **kwargs,
                 ):
        super(ICMPPOAgent, self).__init__(state_size=state_size,
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
        
        self.icm = Network(icm_network, state_size, action_size, eta, self.action_type).to(self.device)

        self.beta = beta
        self.lamb = lamb
        self.eta = eta
        self.extrinsic_coeff = extrinsic_coeff
        self.intrinsic_coeff = intrinsic_coeff

    def learn(self):
        transitions = self.memory.rollout()
        state, action, reward, next_state, done = map(lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device), transitions)
        
        #ICM 
        r_i, _, _ = self.icm(state, action, next_state)
        reward = self.extrinsic_coeff * reward + self.intrinsic_coeff * r_i.unsqueeze(1)
        
        # set log_pi_old and advantage
        with torch.no_grad():            
            if self.action_type == "continuous":
                mu, std, value = self.network(state)
                m = Normal(mu, std)
                z = torch.atanh(torch.clamp(action, -1+1e-7, 1-1e-7))
                log_pi = m.log_prob(z)
                log_pi -= torch.log(1 - action.pow(2) + 1e-7)
            else:
                pi, value = self.network(state)
                log_pi = torch.log(pi.gather(1, action.long()))
            log_pi_old = log_pi
            
            next_value = self.network(next_state)[-1]
            delta = reward + (1 - done) * self.gamma * next_value - value
            adv = delta.clone() 
            for t in reversed(range(len(adv))):
                if t > 0 and (t + 1) % self.n_step == 0:
                    continue
                adv[t] += (1 - done[t]) * self.gamma * self._lambda * adv[t+1]
            adv = (adv - adv.mean()) / (adv.std() + 1e-7)
            ret = adv + value
        
        # start train iteration
        idxs = np.arange(len(reward))
        for _ in range(self.n_epoch):
            np.random.shuffle(idxs)
            for start in range(0, len(reward), self.batch_size):
                end = start + self.batch_size
                idx = idxs[start:end]
                
                _state, _action, _ret, _next_state, _done, _adv, _log_pi_old =\
                    map(lambda x: x[idx], [state, action, ret, next_state, done, adv, log_pi_old])

                if self.action_type == "continuous":
                    mu, std, value = self.network(_state)
                    try:
                        m = Normal(mu, std)
                    except Exception as e:
                        print(log_pi_old.min().item())
                    z = torch.atanh(torch.clamp(_action, -1+1e-7, 1-1e-7))
                    log_pi = m.log_prob(z)
                    log_pi -= torch.log(1 - _action.pow(2) + 1e-7)
                else:
                    pi, value = self.network(_state)
                    log_pi = torch.log(pi.gather(1, _action.long()))
            
                ratio = (log_pi.exp()/(_log_pi_old.exp() + 1e-4)).prod(1, keepdim=True)
                surr1 = ratio * _adv
                surr2 = torch.clamp(ratio, min=1-self.epsilon_clip, max=1+self.epsilon_clip) * _adv
                actor_loss = -torch.min(surr1, surr2).mean() 
                
                critic_loss = F.mse_loss(value, _ret).mean()
                
                entropy_loss = -(-log_pi).mean()
                
                
                # ICM
                _, l_f, l_i = self.icm(_state, _action, _next_state)
                    
                loss_origin = actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss
                loss = self.lamb * loss_origin + (self.beta * l_f) + ((1-self.beta) * l_i)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.)
                self.optimizer.step()

        result = {
            'actor_loss' : actor_loss.item(),
            'critic_loss' : critic_loss.item(),
            'loss' : loss.item(),
            'r_i' : r_i.mean().item(),
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