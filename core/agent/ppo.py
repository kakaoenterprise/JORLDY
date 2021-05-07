import torch
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np

from .reinforce import REINFORCEAgent

class PPOAgent(REINFORCEAgent):
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
                 **kwargs,
                 ):
        super(PPOAgent, self).__init__(state_size=state_size,
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
        
    def act(self, state, training=True):
        if self.action_type == "continuous":
            mu, std, _ = self.network(torch.FloatTensor(state).to(self.device))
            std = std if training else 0
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
            else:
                pi, _ = self.network(state)
                log_pi = torch.log(pi.gather(1, action.long()))
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

                value = self.network(_state)[-1]
                critic_loss = F.mse_loss(value, _ret).mean()

                if self.action_type == "continuous":
                    mu, std, _ = self.network(_state)
                    m = Normal(mu, std)
                    z = torch.atanh(torch.clamp(_action, -1+1e-7, 1-1e-7))
                    log_pi = m.log_prob(z)
                    log_pi -= torch.log(1 - _action.pow(2) + 1e-7)
                else:
                    pi, _ = self.network(_state)
                    log_pi = torch.log(pi.gather(1, _action.long()))

                ratio = torch.exp(log_pi - _log_pi_old)
                surr1 = ratio * _adv
                surr2 = torch.clamp(ratio, min=1-self.epsilon_clip, max=1+self.epsilon_clip) * _adv
                actor_loss = -torch.min(surr1, surr2).mean() 
                
                entopy_loss = -(-log_pi).mean()
                loss = actor_loss + self.vf_coef * critic_loss + self.ent_coef * entopy_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        result = {
            'actor_loss' : actor_loss.item(),
            'critic_loss' : critic_loss.item(),
        }
        return result

    def process(self, state, action, reward, next_state, done):
        result = None
        # Process per step
        self.memory.store(state, action, reward, next_state, done)

        # Process per epi
        if self.memory.size >= self.n_step :
            result = self.learn()
        
        return result