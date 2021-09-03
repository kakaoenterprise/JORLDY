import torch
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import os
import numpy as np

from .ppo import PPO
from core.network import Network

import torch.optim as optim

class RND_PPO(PPO):
    def __init__(self,
                 state_size,
                 action_size,
                 # Parameters for Random Network Distillation
                 rnd_network="rnd_cnn",
                 gamma_i=0.99,
                 extrinsic_coeff=1.0,
                 intrinsic_coeff=1.0,
                 obs_normalize=True,
                 ri_normalize=True,
                 batch_norm=True,
                 **kwargs,
                 ):
        super(RND_PPO, self).__init__(state_size=state_size,
                                      action_size=action_size,
                                      **kwargs)
        self.gamma_i = gamma_i
        self.extrinsic_coeff = extrinsic_coeff
        self.intrinsic_coeff = intrinsic_coeff
        
        self.obs_normalize = obs_normalize
        self.ri_normalize = ri_normalize
        self.batch_norm = batch_norm
        
        self.rnd = Network(rnd_network, state_size, action_size, self.num_workers,
                           gamma_i, ri_normalize, obs_normalize, batch_norm).to(self.device)

        self.optimizer.add_param_group({'params':self.rnd.parameters()})
        
        # Freeze random network
        for name, param in self.rnd.named_parameters():
            if "target" in name:
                param.requires_grad = False

    def learn(self):
        transitions = self.memory.rollout()
        for key in transitions.keys():
            transitions[key] = torch.as_tensor(transitions[key], dtype=torch.float32, device=self.device)

        state = transitions['state']
        action = transitions['action']
        reward = transitions['reward']
        next_state = transitions['next_state']
        done = transitions['done']
        
        # set pi_old and advantage
        with torch.no_grad():
            # RND: calculate exploration reward, update moments of obs and r_i
            self.rnd.update_rms_obs(next_state)
            r_i = self.rnd(next_state, update_ri=True)
            r_i = r_i.unsqueeze(-1)

            # Scaling extrinsic and intrinsic reward
            reward *= self.extrinsic_coeff
            r_i *= self.intrinsic_coeff

            if self.action_type == "continuous":
                mu, std, value = self.network(state)
                m = Normal(mu, std)
                z = torch.atanh(torch.clamp(action, -1+1e-7, 1-1e-7))
                prob = m.log_prob(z).exp()
            else:
                pi, value = self.network(state)
                prob = pi.gather(1, action.long())
            prob_old = prob
            v_i = self.network.get_vi(state)
            
            next_value = self.network(next_state)[-1]
            next_vi = self.network.get_vi(next_state)
            delta = reward + (1 - done) * self.gamma * next_value - value
            # non-episodic intrinsic reward, hence (1-done) not applied
            delta_i = r_i + self.gamma_i * next_vi - v_i
            adv = delta.clone() 
            adv_i = delta_i.clone()
            adv, adv_i, done = adv.view(-1, self.n_step), adv_i.view(-1, self.n_step), done.view(-1, self.n_step)
            for t in reversed(range(self.n_step - 1)):
                adv[:, t] += (1 - done[:, t]) * self.gamma * self._lambda * adv[:, t+1]
                adv_i[:, t] += self.gamma_i * self._lambda * adv_i[:, t+1]
                
            if self.use_standardization:
                adv = (adv - adv.mean(dim=1, keepdim=True)) / (adv.std(dim=1, keepdim=True) + 1e-7)
                adv_i = (adv_i - adv_i.mean(dim=1, keepdim=True)) / (adv_i.std(dim=1, keepdim=True) + 1e-7)
            adv = adv.view(-1, 1)
            adv_i = adv_i.view(-1, 1)
            done = done.view(-1, 1)
            
            ret = adv + value
            ret_i = adv_i + v_i
        
        # start train iteration
        actor_losses, critic_losses, entropy_losses, rnd_losses, ratios, probs = [], [], [], [], [], []
        idxs = np.arange(len(reward))
        for idx_epoch in range(self.n_epoch):
            np.random.shuffle(idxs)
            for offset in range(0, len(reward), self.batch_size):
                idx = idxs[offset : offset + self.batch_size]
                
                _state, _action, _ret, _next_state, _adv, _prob_old =\
                    map(lambda x: x[idx], [state, action, ret, next_state, adv, prob_old])
                _ret_i, _adv_i = map(lambda x: x[idx], [ret_i, adv_i])

                _r_i = self.rnd.forward(_next_state) * self.intrinsic_coeff
                
                if self.action_type == "continuous":
                    mu, std, value = self.network(_state)
                    m = Normal(mu, std)
                    z = torch.atanh(torch.clamp(_action, -1+1e-7, 1-1e-7))
                    prob = m.log_prob(z).exp()
                else:
                    pi, value = self.network(_state)
                    m = Categorical(pi)
                    prob = pi.gather(1, _action.long())
                _v_i = self.network.get_vi(_state)
                
                ratio = (prob / (_prob_old + 1e-4)).prod(1, keepdim=True)
                surr1 = ratio * (_adv + _adv_i)
                surr2 = torch.clamp(ratio, min=1-self.epsilon_clip, max=1+self.epsilon_clip) * (_adv + _adv_i)
                actor_loss = -torch.min(surr1, surr2).mean() 
                
                critic_loss = F.mse_loss(value, _ret).mean() + F.mse_loss(_v_i, _ret_i).mean()
                
                entropy_loss = -m.entropy().mean()
                ppo_loss = actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss
                rnd_loss = _r_i.mean()
                
                loss = ppo_loss + rnd_loss
                
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.rnd.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                
                probs.append(prob.min().item())
                ratios.append(ratio.max().item())
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy_loss.item())
                rnd_losses.append(rnd_loss.item())
            
        result = {
            'actor_loss' : np.mean(actor_losses),
            'critic_loss' : np.mean(critic_losses),
            'entropy_loss' : np.mean(entropy_losses),
            'r_i': np.mean(rnd_losses),
            'max_ratio' : max(ratios),
            'min_prob': min(probs),
            'min_prob_old': prob_old.min().item(),
        }
        return result

    def save(self, path):
        print(f"...Save model to {path}...")
        torch.save({
            "network" : self.network.state_dict(),
            "rnd" : self.rnd.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
        }, os.path.join(path,"ckpt"))

    def load(self, path):
        print(f"...Load model from {path}...")
        checkpoint = torch.load(os.path.join(path,"ckpt"),map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.rnd.load_state_dict(checkpoint["rnd"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
