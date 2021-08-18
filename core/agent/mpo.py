import torch
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np
import os, copy

from .base import BaseAgent
from core.network import Network
from core.optimizer import Optimizer
from core.buffer import MPOBuffer

class MPO(BaseAgent):
    def __init__(self,
                 state_size,
                 action_size,
                 optim_config={'name': 'adam'},
                 network="discrete_policy_q",
                 buffer_size=50000,
                 batch_size=64,
                 start_train_step=2000,
                 target_update_period=500,
                 n_step=100,
                 clip_grad_norm=1.0,
                 gamma=0.99,
                 device=None,
                 # parameters unique to MPO
                 num_sample = 30,
                 min_eta=1e-8,
                 min_alpha_mu=1e-8,
                 min_alpha_sigma=1e-8,
                 eps_eta=0.1,
                 eps_alpha_mu=0.1,
                 eps_alpha_sigma=0.1,
                 eta=0.1, 
                 alpha_mu=0.1,
                 alpha_sigma=0.1,
                 ):

        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_type = network.split("_")[0]
        assert self.action_type in ["continuous", "discrete"]
        self.action_size = action_size

        self.network = Network(network, state_size, action_size).to(self.device)
        self.target_network = copy.deepcopy(self.network)
        
        self.batch_size = batch_size
        self.n_step = n_step
        self.clip_grad_norm = clip_grad_norm

        self.num_learn = 0
        self.time_t = 0
        self.start_train_step = start_train_step
        self.target_update_stamp = 0
        self.target_update_period = target_update_period
        
        self.num_sample = num_sample
        
        self.ones = None
        
        self.min_eta = torch.tensor(min_eta, device=self.device)
        self.min_alpha_mu = torch.tensor(min_alpha_mu, device=self.device)
        self.min_alpha_sigma = torch.tensor(min_alpha_sigma, device=self.device)
        
        self.eps_eta = eps_eta
        self.eps_alpha_mu = eps_alpha_mu
        self.eps_alpha_sigma = eps_alpha_sigma
        
        self.eta = torch.nn.Parameter(torch.tensor(eta, requires_grad=True).to(self.device))
        self.alpha_mu = torch.nn.Parameter(torch.tensor(alpha_mu, requires_grad=True).to(self.device))
        self.alpha_sigma = torch.nn.Parameter(torch.tensor(alpha_sigma, requires_grad=True).to(self.device))

        self.reset_lgr_muls()
        
        self.action_type = network.split("_")[0]
        assert self.action_type in ["continuous", "discrete"]

        self.optimizer = Optimizer(**optim_config, params=list(self.network.parameters()) + [self.eta, self.alpha_mu, self.alpha_sigma])

        self.gamma = gamma
        self.memory = MPOBuffer(buffer_size, self.n_step)
        
    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training)
        if self.action_type == "continuous":
            mu, std = self.network(torch.as_tensor(state, dtype=torch.float32, device=self.device))

            m = Normal(mu, std)
            z = m.sample() if training else mu
            action = torch.tanh(z)
            action = action.data.cpu().numpy()
            prob = m.log_prob(z).sum(axis=-1, keepdims=True)
            prob = prob.exp().cpu().numpy()
                
        else:
            pi, _ = self.network(torch.as_tensor(state, dtype=torch.float32, device=self.device))
            action = torch.multinomial(pi, 1) if training else torch.argmax(pi, dim=-1, keepdim=True)
            prob = pi.numpy()
            action = action.cpu().numpy()
        return {
            'action': action,
            'prob': prob,
        }

    def learn(self):
        transitions = self.memory.sample(self.batch_size)
        for key in transitions.keys():
            transitions[key] = torch.as_tensor(transitions[key], dtype=torch.float32, device=self.device)
            
        state = transitions['state']
        action = transitions['action']
        reward = transitions['reward']
        next_state = transitions['next_state']
        done = transitions['done']
        prob_b = transitions['prob']
        
        if self.action_type == "continuous":
            with torch.no_grad():
                mu, std = self.network(state)
                
                m = Normal(mu, std)
                z = torch.atanh(torch.clamp(action, -1+1e-7, 1-1e-7))
                log_pi = m.log_prob(z)
                log_pi = log_pi.sum(axis=-1, keepdims=True)
                
                mu_old = mu
                std_old = std
                pi_old = torch.exp(log_pi)
                
                Qt_a = self.target_network.calculate_Q(state, action)
                
                next_mu, next_std = self.network(next_state)
                m = Normal(next_mu, next_std)
                z = m.sample((self.num_sample,)) # (num_sample, batch_size, len_tr, dim_action)
                next_action = torch.tanh(z)
                
                Qt_next = self.target_network.calculate_Q(next_state.unsqueeze(0).repeat(self.num_sample, 1, 1, 1), next_action) # (num_sample, batch_size, len_tr, 1)
                
                prob_t = pi_old
                
                # calculate Qret for Retrace
                if self.ones == None: self.ones = torch.ones(prob_t.shape).to(self.device)
                c = torch.min(self.ones, prob_t / prob_b)
                
                Qret = reward + Qt_next.mean(axis=0) - Qt_a
                for i in reversed(range(reward.shape[1]-1)):
                    Qret[:, i] += self.gamma * c[:, i+1] * Qret[:, i+1] * (1-done[:,i])
                
                Qret += Qt_a
                
            mu, std = self.network(state)
            
            Q = self.network.calculate_Q(state, action)
            m = Normal(mu, std)
            z = torch.atanh(torch.clamp(action, -1+1e-7, 1-1e-7))
            log_pi = m.log_prob(z)
            log_pi = log_pi.sum(axis=-1, keepdims=True)
            pi = torch.exp(log_pi)
            
            z_add = m.sample((self.num_sample, )) # (num_sample, batch_size, len_tr, dim_action)
            action_add = torch.tanh(z_add)
            log_pi_add = m.log_prob(z_add)
            log_pi_add = log_pi_add.sum(axis=-1, keepdims=True)
            Q_add = self.target_network.calculate_Q(state.unsqueeze(0).repeat(self.num_sample, 1, 1, 1), action_add)
            
            critic_loss = F.mse_loss(Q, Qret).mean()
            
            exp_Q_eta = torch.exp(Q.detach() / self.eta)
            exp_Q_eta_add = torch.exp(Q_add.detach() / self.eta)
            q = torch.softmax(exp_Q_eta_add, axis = 0)
            
            actor_loss = -torch.mean(q.detach() * log_pi_add)
            
            eta_loss = self.eta * self.eps_eta + \
                       self.eta * torch.mean(torch.log(exp_Q_eta_add.mean(axis=0)))
            
            ss = 1. / (std**2) # (batch_size, len_tr, action_dim)
            ss_old = 1. / (std_old ** 2)

            # mu
            d_mu = mu - mu_old.detach() # (batch_size, len_tr, action_dim)
            KLD_mu = 0.5 * torch.sum(d_mu* 1./ss_old.detach() * d_mu, axis = -1)
            mu_loss = torch.mean(self.alpha_mu * (self.eps_alpha_mu - KLD_mu.detach()) + \
                                 self.alpha_mu.detach() * KLD_mu)

            # sigma
            KLD_sigma = 0.5 * (torch.sum(1./ss * ss_old.detach(), axis = -1) - \
                               ss.shape[-1] + \
                               torch.log(torch.prod(ss, axis = -1)/torch.prod(ss_old.detach(), axis = -1)))
            sigma_loss = torch.mean(self.alpha_sigma * (self.eps_alpha_sigma - KLD_sigma.detach()) + \
                                 self.alpha_sigma.detach() * KLD_sigma)

            alpha_loss = mu_loss + sigma_loss
                
        else:
            with torch.no_grad():
                # calculate Q_ret using Retrace
                _, Qt = self.target_network(state) # Q_target
                _, Qt_next = self.target_network(next_state)
                pi, _ = self.network(state)
                pi_next, _ = self.network(next_state)

                Qt_a = Qt.gather(2, action.long()) # (batch_size, len_tr, 1)
                prob_t = pi.gather(2, action.long()) # (batch_size, len_tr, 1), target policy probability
                
                if self.ones == None: self.ones = torch.ones(prob_t.shape).to(self.device)

                c = torch.min(self.ones, prob_t/prob_b.gather(2, action.long())) # (batch_size, len_tr, 1), prod of importance ratio and gamma
                Qret = reward + torch.sum(pi_next * Qt_next, axis=2).unsqueeze(2) - Qt_a
                for i in reversed(range(reward.shape[1]-1)): # along the trajectory length
                    Qret[:, i] += self.gamma * c[:, i+1] * Qret[:, i+1] * (1-done[:,i])

                Qret += Qt_a
                pi_old = pi
                
            pi, Q = self.network(state) # pi,Q: (batch_size, len_tr, dim_action)
            Q_a = Q.gather(2, action.long())
            critic_loss = F.mse_loss(Q_a, Qret).mean()
            
            exp_Q_eta = torch.exp(Q / self.eta)
            q = exp_Q_eta / torch.sum(exp_Q_eta.detach(), axis = 2, keepdims=True)
            
            actor_loss = -torch.mean(q.detach() * torch.log(pi))
            
            eta_loss = self.eta * self.eps_eta + \
                       self.eta * torch.mean(torch.log(torch.sum(pi * exp_Q_eta, axis=2)))
            
            KLD_pi = pi_old.detach() * (torch.log(pi_old.detach()) - torch.log(pi))
            KLD_pi = torch.sum(KLD_pi, axis = len(pi_old.shape)-1)
            alpha_loss = torch.mean(self.alpha_mu * (self.eps_alpha_mu - KLD_pi.detach()) + \
                                    self.alpha_mu.detach() * KLD_pi)

        loss = critic_loss + actor_loss + eta_loss + alpha_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        self.reset_lgr_muls()
        
        self.num_learn += 1

        result = {
            'actor_loss' : actor_loss.item(),
            'critic_loss' : critic_loss.item(),
            'eta_loss': eta_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'eta': self.eta.item(),
            'alpha_mu': self.alpha_mu.item(),
            'alpha_sigma': self.alpha_sigma.item(),
            'max_probt': prob_t.cpu().numpy().max(),
            'max_probb': prob_b.cpu().numpy().max(),
            'min_Q': Q.detach().cpu().numpy().min(),
            'max_Q': Q.detach().cpu().numpy().max(),
        }
        
        if self.action_type == 'continuous':
            result['min_mu'] = mu.detach().cpu().numpy().min()
            result['max_mu'] = mu.detach().cpu().numpy().max()
            result['min_std'] = std.detach().cpu().numpy().min()
            result['max_std'] = std.detach().cpu().numpy().max()
            
        return result
    
    # reset Lagrange multipliers: eta, alpha_{mu, sigma}
    def reset_lgr_muls(self):
        self.eta.data = torch.max(self.eta, self.min_eta)
        self.alpha_mu.data = torch.max(self.alpha_mu, self.min_alpha_mu)
        self.alpha_sigma.data = torch.max(self.alpha_sigma, self.min_alpha_sigma)
        
    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())
        
    def save(self, path):
        print(f"...Save model to {path}...")
        torch.save({
            "network" : self.network.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
        }, os.path.join(path,"ckpt"))
        
    def load(self, path):
        print(f"...Load model from {path}...")
        checkpoint = torch.load(os.path.join(path,"ckpt"),map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.target_network = copy.deepcopy(self.network)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        
    def process(self, transitions, step):
        result = {}
        
        # Process per step
        self.memory.store(transitions)
        delta_t = step - self.time_t
        self.time_t = step
        self.target_update_stamp += delta_t
        
        if self.memory.size > self.batch_size and self.time_t >= self.start_train_step:
            result = self.learn()
            
        if self.num_learn > 0 and \
           self.target_update_stamp > self.target_update_period:
            self.update_target()
            self.target_update_stamp = 0
        
        return result