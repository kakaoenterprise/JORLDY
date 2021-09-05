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
                 actor="discrete_policy",
                 critic="dqn",
                 head = 'mlp',
                 buffer_size=50000,
                 batch_size=64,
                 start_train_step=2000,
                 n_epoch=64,
                 n_step=8,
                 clip_grad_norm=1.0,
                 gamma=0.99,
                 device=None,
                 num_workers=1,
                 # parameters unique to MPO
                 critic_loss_type = 'retrace', # one of ['1-step TD', 'retrace']
                 num_sample=30,
                 min_eta=1e-8,
                 min_alpha_mu=1e-8,
                 min_alpha_sigma=1e-8,
                 eps_eta=0.01,
                 eps_alpha_mu=0.01,
                 eps_alpha_sigma=5*1e-5,
                 eta=1.0, 
                 alpha_mu=1.0,
                 alpha_sigma=1.0,
                 **kwargs,
                 ):

        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head = head
        self.action_type = actor.split("_")[0]
        assert self.action_type in ["continuous", "discrete"]
        self.action_size = action_size

        self.actor = Network(actor, state_size, action_size, head=head).to(self.device)
        self.target_actor = Network(actor, state_size, action_size, head=head).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic_loss_type = critic_loss_type
        assert self.critic_loss_type in ['1-step TD', 'retrace']
        self.critic = Network(critic, state_size, action_size, head=head).to(self.device)
        self.target_critic = Network(critic, state_size, action_size, head=head).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.batch_size = batch_size
        self.n_step = n_step
        self.clip_grad_norm = clip_grad_norm

        self.num_learn = 0
        self.time_t = 0
        self.start_train_step = start_train_step
        self.n_epoch = n_epoch
        
        self.num_sample = num_sample
        
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

        self.actor_optimizer = Optimizer(**optim_config, params=list(self.actor.parameters()) + [self.eta, self.alpha_mu, self.alpha_sigma])
        self.critic_optimizer = Optimizer(**optim_config, params=list(self.critic.parameters()))

        self.gamma = gamma
        self.memory = MPOBuffer(buffer_size, n_step, num_workers)
        
    @torch.no_grad()
    def act(self, state, training=True):
        self.actor.train(training)
        if self.action_type == "continuous":
            mu, std = self.actor(torch.as_tensor(state, dtype=torch.float32, device=self.device))

            m = Normal(mu, std)
            z = m.sample() if training else mu
            action = torch.tanh(z)
            action = action.data.cpu().numpy()
            prob = m.log_prob(z).sum(axis=-1, keepdims=True)
            prob = prob.exp().cpu().numpy()
                
        else:
            pi = self.actor(torch.as_tensor(state, dtype=torch.float32, device=self.device))
            action = torch.multinomial(pi, 1) if training else torch.argmax(pi, dim=-1, keepdim=True)
            action = action.cpu().numpy()
            prob = np.take(pi.cpu().numpy(), action)
        return {
            'action': action,
            'prob': prob,
        }

    def learn(self):
            
        transitions = self.memory.sample(self.batch_size)
        for key in transitions.keys():
            # reshape: (batch_size, len_tr, item_dim)
            #        -> (batch_size * len_tr, item_dim)
            transitions[key] = torch.as_tensor(transitions[key], dtype=torch.float32, device=self.device).view(-1, *transitions[key].shape[2:])
            
        state = transitions['state']
        action = transitions['action']
        reward = transitions['reward']
        next_state = transitions['next_state']
        done = transitions['done']
        prob_b = transitions['prob']
        
        if self.action_type == "continuous":
            
            mu, std = self.actor(state)
            Q = self.critic(state, action)
            m = Normal(mu, std)
            z = torch.atanh(torch.clamp(action, -1+1e-7, 1-1e-7))
            log_pi = m.log_prob(z)
            log_prob = log_pi.sum(axis=-1, keepdims=True)
            prob = torch.exp(log_prob)
            
            with torch.no_grad():
                mut, stdt = self.target_actor(state)
                
                mt = Normal(mut, stdt)
                zt = torch.atanh(torch.clamp(action, -1+1e-7, 1-1e-7))
                log_pit = mt.log_prob(zt)
                log_probt = log_pit.sum(axis=-1, keepdims=True)
                
                mu_old = mut
                std_old = stdt
                prob_t = torch.exp(log_probt)
                
                Qt_a = self.target_critic(state, action)
                
                next_mu, next_std = self.actor(next_state)
                mn = Normal(next_mu, next_std)
                zn = mn.sample((self.num_sample,)) # (num_sample, batch_size * len_tr, dim_action)
                next_action = torch.tanh(zn)
                
                Qt_next = self.target_critic(next_state.unsqueeze(0).repeat(self.num_sample, 1, 1), next_action) # (num_sample, batch_size * len_tr, 1)
                
                c = torch.clip(prob/(prob_b+1e-6), max=1.)
                
                if self.critic_loss_type == '1-step TD':
                    Qret = reward + self.gamma * (1-done) * Qt_next.mean(axis=0)
                elif self.critic_loss_type == 'retrace':
                    Qret = reward + self.gamma * Qt_next.mean(axis=0) * (1-done)

                    # temporarily reshaping values
                    # (batch_size * len_tr, item_dim) -> (batch_size, len_tr, item_dim)
                    Qret = Qret.view(self.batch_size, -1, *Qret.shape[1:])
                    Qt_a = Qt_a.view(self.batch_size, -1, *Qt_a.shape[1:])
                    c = c.view(self.batch_size, -1, *c.shape[1:])
                    done = done.view(self.batch_size, -1, *done.shape[1:])
                    for i in reversed(range(Qret.shape[1]-1)):
                        Qret[:, i] += self.gamma * c[:, i+1] * (1-done[:,i]) * (Qret[:, i+1] - Qt_a[:, i+1])
                    Qret = Qret.view(-1, *Qret.shape[2:])
        
            zt_add = mt.sample((self.num_sample, )) # (num_sample, batch_size * len_tr, dim_action)
            action_add = torch.tanh(zt_add)
            log_pi_add = m.log_prob(zt_add)
            log_prob_add = log_pi_add.sum(axis=-1, keepdims=True)
            Qt_add = self.target_critic(state.unsqueeze(0).repeat(self.num_sample, 1, 1), action_add)
            
            critic_loss = F.mse_loss(Q, Qret).mean()
            
            # Calculate Vt_add, At_add using Qt_add
            Vt_add = torch.mean(Qt_add, axis=0, keepdims=True)
            At_add = Qt_add - Vt_add
            At = At_add
            
            ''' variational distribution q uses exp(At / eta) instead of exp(Qt / eta), for stable learning'''
            q = torch.softmax(At_add / self.eta, axis = 0)
            actor_loss = -torch.mean(torch.sum(q.detach() * log_prob_add, axis=0))

            eta_loss = self.eta * self.eps_eta + \
                       self.eta * torch.mean(torch.log(torch.exp((At_add) / self.eta).mean(axis=0)))
            
            ss = 1. / (std**2) # (batch_size * len_tr, action_dim)
            ss_old = 1. / (std_old ** 2)

            '''
            KL-Divergence losses(related to alpha) implemented using methods introduced from V-MPO paper
            https://arxiv.org/abs/1909.12238
            '''
            
            # mu
            d_mu = mu - mu_old.detach() # (batch_size * len_tr, action_dim)
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
            
            pi = self.actor(state) # pi,Q: (batch_size, len_tr, dim_action)
            pi_next = self.actor(next_state)
            Q = self.critic(state)
            Q_a = Q.gather(1, action.long())
                
            with torch.no_grad():
                # calculate Q_ret using Retrace
                Qt = self.target_critic(state) # Q_target
                Qt_next = self.target_critic(next_state)
                pit = self.target_actor(state)
                
                Qt_a = Qt.gather(1, action.long())
                prob_t = pi.gather(1, action.long()) # (batch_size * len_tr, 1), target policy probability
                
                c = torch.clip(prob_t/(prob_b+1e-6), max=1.)# (batch_size * len_tr, 1), prod of importance ratio and gamma

                if self.critic_loss_type == '1-step TD':
                    Qret = reward + self.gamma * (1-done) * torch.sum(pi_next * Qt_next, axis=-1, keepdim=True)
                elif self.critic_loss_type == 'retrace':
                    Qret = reward + self.gamma * torch.sum(pi_next * Qt_next, axis=-1, keepdim=True) * (1-done)

                    # temporarily reshaping values
                    # (batch_size * len_tr, item_dim) -> (batch_size, len_tr, item_dim)
                    Qret = Qret.view(self.batch_size, -1, *Qret.shape[1:])
                    Qt_a = Qt_a.view(self.batch_size, -1, *Qt_a.shape[1:])
                    c = c.view(self.batch_size, -1, *c.shape[1:])
                    done = done.view(self.batch_size, -1, *done.shape[1:])
                    for i in reversed(range(Qret.shape[1]-1)): # along the trajectory length
                        Qret[:, i] += self.gamma * c[:, i+1] * (Qret[:, i+1] - Qt_a[:, i+1]) * (1-done[:,i])
                    Qret = Qret.view(-1, *Qret.shape[2:])
                
                pi_old = pit
                
            critic_loss = F.mse_loss(Q_a, Qret).mean()

            # calculate V, Advantage of Qt
            Vt = torch.sum(pi_old * Qt, axis=-1, keepdims=True)
            At = Qt - Vt
            
            ''' variational distribution q uses exp(At / eta) instead of exp(Qt / eta), for stable learning'''
            q = torch.softmax(At / self.eta, axis=-1)    
            actor_loss = -torch.mean(torch.sum(q.detach() * torch.log(pi), axis=-1))
            
            eta_loss = self.eta * self.eps_eta + \
                       self.eta * torch.mean(torch.log(torch.sum(pi_old * torch.exp(At / self.eta), axis=-1)))
            
            '''
            KL-Divergence losses(related to alpha) implemented using methods introduced from V-MPO paper
            https://arxiv.org/abs/1909.12238
            '''
            
            KLD_pi = pi_old.detach() * (torch.log(pi_old.detach()) - torch.log(pi))
            KLD_pi = torch.sum(KLD_pi, axis = len(pi_old.shape)-1)
            alpha_loss = torch.mean(self.alpha_mu * (self.eps_alpha_mu - KLD_pi.detach()) + \
                                    self.alpha_mu.detach() * KLD_pi)

        loss = critic_loss + actor_loss + eta_loss + alpha_loss

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()
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
            'min_Q': Q.detach().cpu().numpy().min(),
            'max_Q': Q.detach().cpu().numpy().max(),
            'min_At': At.detach().cpu().numpy().min(),
            'max_At': At.detach().cpu().numpy().max(),
        }
            
        return result
    
    # reset Lagrange multipliers: eta, alpha_{mu, sigma}
    def reset_lgr_muls(self):
        self.eta.data = torch.max(self.eta, self.min_eta)
        self.alpha_mu.data = torch.max(self.alpha_mu, self.min_alpha_mu)
        self.alpha_sigma.data = torch.max(self.alpha_sigma, self.min_alpha_sigma)
        
    def update_target(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
    def save(self, path):
        print(f"...Save model to {path}...")
        torch.save({
            "actor" : self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer" : self.actor_optimizer.state_dict(),
            "critic_optimizer" : self.critic_optimizer.state_dict(),
        }, os.path.join(path,"ckpt"))
        
    def load(self, path):
        print(f"...Load model from {path}...")
        checkpoint = torch.load(os.path.join(path,"ckpt"),map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        
    def process(self, transitions, step):
        result = {}
        
        # Process per step
        self.memory.store(transitions)
        delta_t = step - self.time_t
        self.time_t = step
        
        if self.memory.size >= self.batch_size and self.time_t >= self.start_train_step:
            for i in range(self.n_epoch):
                result = self.learn()
            self.update_target()
        
        return result
    
    def sync_in(self, weights):
        self.actor.load_state_dict(weights)
    
    def sync_out(self, device="cpu"):
        weights = self.actor.state_dict()
        for k, v in weights.items():
            weights[k] = v.to(device) 
        sync_item ={
            "weights": weights,
        }
        return sync_item

