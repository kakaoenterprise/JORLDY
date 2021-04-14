import torch
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np
import os

from core.network import Network
from core.optimizer import Optimizer
from .utils import ReplayBuffer
from .reinforce import REINFORCEAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        
    def act(self, state, training=True):
        if self.action_type == "continuous":
            mu, std, _ = self.network(torch.FloatTensor(state).to(device))
            std = std if training else 0
            m = Normal(mu, std)
            z = m.sample()
            action = torch.tanh(z)
            action = action.data.cpu().numpy()
        else:
            pi, _ = self.network(torch.FloatTensor(state).to(device))
            m = Categorical(pi)
            action = m.sample().data.cpu().numpy()[..., np.newaxis]
        return action

    def learn(self):
        transitions = self.memory.rollout()
        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device), transitions)
        
        # set advantage and log_pi_old
        with torch.no_grad():
            value = self.network(state)[-1]
            next_value = self.network(next_state)[-1]
            target_value = reward + (1 - done) * self.gamma * next_value
            advantage = target_value - value
            for t in reversed(range(len(advantage))):
                if t > 0 and (t + 1) % self.n_step == 0:
                    continue
                advantage[t] += (1 - done[t]) * self.gamma * self._lambda * advantage[t+1]
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-7)
            
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
                
                _state, _action, _reward, _next_state, _done, _advantage, _log_pi_old =\
                    map(lambda x: x[idx], [state, action, reward, next_state, done, advantage, log_pi_old])

                with torch.no_grad():
                    next_value = self.network(_next_state)[-1]
                    target_value = _reward + (1 - _done) * self.gamma * next_value
                value = self.network(_state)[-1]
                critic_loss = F.mse_loss(value, target_value).mean()

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
                surr1 = ratio * _advantage
                surr2 = torch.clamp(ratio, min=1-self.epsilon_clip, max=1+self.epsilon_clip) * _advantage
                actor_loss = -torch.min(surr1, surr2).mean() 

                loss = actor_loss + critic_loss

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