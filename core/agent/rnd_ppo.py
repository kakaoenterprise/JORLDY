import torch
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np

from .reinforce import REINFORCEAgent
from core.optimizer import Optimizer
from core.network import Network

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
        
        self.rnd = Network(rnd_network, state_size, action_size).to(self.device)
        self.rnd_optimizer = Optimizer('adam', self.rnd.parameters(), lr=self.learning_rate)
        
        # Freeze random network
        for name, param in self.rnd.named_parameters():
            if "target" in name:
                param.requires_grad = False
        
        self.gamma_i = gamma_i
        self.extrinsic_coeff = extrinsic_coeff
        self.intrinsic_coeff = intrinsic_coeff
                
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
        actor_losses, critic_losses, entropy_losses, ratios, pis, pi_olds = [], [], [], [], [], []
        pi_olds.append(pi_old.min().item())
        idxs = np.arange(len(reward))
        for _ in range(self.n_epoch):
            np.random.shuffle(idxs)
            for offset in range(0, len(reward), self.batch_size):
                idx = idxs[offset : offset + self.batch_size]
                
                _state, _action, _ret, _next_state, _done, _adv, _pi_old =\
                    map(lambda x: x[idx], [state, action, ret, next_state, done, adv, pi_old])

                if self.action_type == "continuous":
                    mu, std, value = self.network(_state)
                    m = Normal(mu, std)
                    z = torch.atanh(torch.clamp(_action, -1+1e-7, 1-1e-7))
                    pi = m.log_prob(z).exp()
                else:
                    pi, value = self.network(_state)
                    pi = pi.gather(1, _action.long())
                
                ratio = (pi / (_pi_old + 1e-4)).prod(1, keepdim=True)
                surr1 = ratio * _adv
                surr2 = torch.clamp(ratio, min=1-self.epsilon_clip, max=1+self.epsilon_clip) * _adv
                actor_loss = -torch.min(surr1, surr2).mean() 
                
                critic_loss = F.mse_loss(value, _ret).mean()
                
                entropy_loss = torch.log(pi + 1e-4).mean()
                loss = actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
                self.optimizer.step()
                
                pis.append(pi.min().item())
                ratios.append(ratio.max().item())
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy_loss.item())
        result = {
            'actor_loss' : np.mean(actor_losses),
            'critic_loss' : np.mean(critic_losses),
            'entropy_loss' : np.mean(entropy_losses),
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