import torch
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np

from .reinforce import REINFORCEAgent
from core.network import Network
from core.optimizer import Optimizer
from .utils import Rollout

class VMPOAgent(REINFORCEAgent):
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
                 min_eta=1e-8,
                 min_alpha_mu=1e-8,
                 min_alpha_sigma = 1e-8,
                 eps_eta=0.1,
                 eps_alpha_mu=0.1,
                 eps_alpha_sigma=0.1,
                 eta=0.1, 
                 alpha_mu=0.1,
                 alpha_sigma=0.1,
                 optimizer="adam",
                 learning_rate=1e-4,
                 gamma=0.99,
                 device=None,
                 ):

        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = batch_size
        self.n_step = n_step
        self.n_epoch = n_epoch
        self._lambda = _lambda
        self.epsilon_clip = epsilon_clip
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.time_t = 0
        self.learn_stamp = 0
        
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

        self.network = Network(network, state_size, action_size).to(self.device)
        self.optimizer = Optimizer(optimizer, list(self.network.parameters()) + 
                                   [self.eta, self.alpha_mu, self.alpha_sigma], lr=learning_rate)

        self.gamma = gamma
        self.memory = Rollout()
        
    def act(self, state, training=True):
        if self.action_type == "continuous":
#             mu, std, _ = self.network(torch.FloatTensor(state).to(self.device))
#             std = std if training else 0
            mu, std, _ = self.network(torch.FloatTensor(state).to(self.device))
            std = std if training else torch.zeros_like(std, device=self.device) + 1e-4

            m = Normal(mu, std)
            z = m.sample()
            action = torch.tanh(z)
            action = action.data.cpu().numpy()
        else:
            pi, _ = self.network(torch.FloatTensor(state).to(self.device))
            m = Categorical(pi)
            action = m.sample().data.cpu().numpy()[..., np.newaxis]
        return action

    def learn(self):
        transitions = self.memory.rollout()
        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(self.device), transitions)
        
        # set advantage and log_pi_old
        with torch.no_grad():
            value = self.network(state)[-1]
            next_value = self.network(next_state)[-1]
            delta = reward + (1 - done) * self.gamma * next_value - value
            adv = delta.clone() 
            for t in reversed(range(len(adv))):
                if t > 0 and (t + 1) % self.n_step == 0:
                    continue
                adv[t] += (1 - done[t]) * self.gamma * self._lambda * adv[t+1]
            adv = (adv - adv.mean()) / (adv.std() + 1e-7)
            ret = adv + value
            
            if self.action_type == "continuous":
                mu, std, _ = self.network(state)
                m = Normal(mu, std)
                z = torch.atanh(torch.clamp(action, -1+1e-7, 1-1e-7))
                log_pi = m.log_prob(z)
                log_pi -= torch.log(1 - action.pow(2) + 1e-7)
                log_pi = log_pi.sum(axis=1,keepdim=True)
                mu_old = mu
                std_old = std
            else:
                pi, _ = self.network(state)
                pi_old = pi
                log_pi = torch.log(pi.gather(1, action.long()))
                log_piall_old = torch.log(pi)

            log_pi_old = log_pi
        
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
                    _mu_old, _std_old = map(lambda x: x[idx], [mu_old, std_old])
                else: 
                    _log_piall_old, _pi_old = map(lambda x: x[idx], [log_piall_old, pi_old])

                # select top 50% of advantages
                idx_tophalf = _adv > _adv.median()
                tophalf_adv = _adv[idx_tophalf]
                # calculate psi
                exp_adv_eta = torch.exp(tophalf_adv / self.eta)
                psi = exp_adv_eta / torch.sum(exp_adv_eta.detach()) # TODO: is it right to detach() the denominator here?

                value = self.network(_state)[-1]
                critic_loss = F.mse_loss(value, _ret).mean()
                
                # calculate loss for eta
                eta_loss = self.eta * self.eps_eta + torch.log(torch.mean(exp_adv_eta))

                if self.action_type == "continuous":
                    mu, std, _ = self.network(_state)
                    m = Normal(mu, std)
                    z = torch.atanh(torch.clamp(_action, -1+1e-7, 1-1e-7))
                    log_pi = m.log_prob(z)
                    log_pi -= torch.log(1 - _action.pow(2) + 1e-7)
                    log_pi = log_pi.sum(axis=1, keepdim=True)
                else:
                    pi, _ = self.network(_state)
                    log_pi = torch.log(pi.gather(1, _action.long()))
                    log_piall = torch.log(pi)

                # calculate policy loss (actor_loss)
                actor_loss = -torch.sum(psi.detach() * log_pi[idx_tophalf])

                # calculate loss for alpha
                # NOTE: assumes that std are in the same shape as mu (hence vectors)
                #       hence each dimension of Gaussian distribution is independent
                if self.action_type == "continuous":
                    ss = 1. / (std**2) # (batch_size * action_dim)
                    ss_old = 1. / (_std_old ** 2) # (batch_size * action_dim)

                    # mu
                    d_mu = mu - _mu_old.detach() # (batch_size * action_dim)
                    KLD_mu = 0.5 * torch.sum(d_mu* 1./ss_old.detach() * d_mu, axis = 1)
                    mu_loss = torch.mean(self.alpha_mu * (self.eps_alpha_mu - KLD_mu.detach()) + \
                                         self.alpha_mu.detach() * KLD_mu)

                    # sigma
                    KLD_sigma = 0.5 * ((torch.sum(1./ss * ss_old.detach(), axis = 1) - ss.shape[-1] + torch.log(torch.prod(ss, axis = 1)/torch.prod(ss_old.detach(), axis = 1))))
                    sigma_loss = torch.mean(self.alpha_sigma * (self.eps_alpha_sigma - KLD_sigma.detach()) + \
                                         self.alpha_sigma.detach() * KLD_sigma)

                    alpha_loss = mu_loss + sigma_loss
                else:
                    KLD_pi = _pi_old.detach() * (_log_pi_old.detach() - log_pi)
                    KLD_pi = torch.sum(KLD_pi, axis = len(_pi_old.shape)-1) # TODO: need to sum over all the possible state-action pairs
                    alpha_loss = torch.mean(self.alpha_mu * (self.eps_alpha_mu - KLD_pi.detach()) + \
                                            self.alpha_mu.detach() * KLD_pi)

                loss = critic_loss + actor_loss + eta_loss + alpha_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.reset_lgr_muls()

        result = {
            'actor_loss' : actor_loss.item(),
            'critic_loss' : critic_loss.item(),
            'eta_loss': eta_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'eta': self.eta.item(),
            'alpha_mu': self.alpha_mu.item(),
            'alpha_sigma': self.alpha_sigma.item(),
        }
        return result
    
    # reset Lagrange multipliers: eta, alpha_{mu, sigma}
    def reset_lgr_muls(self):
        self.eta.data = torch.max(self.eta, self.min_eta)
        self.alpha_mu.data = torch.max(self.alpha_mu, self.min_alpha_mu)
        self.alpha_sigma.data = torch.max(self.alpha_sigma, self.min_alpha_sigma)
        
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