import torch
torch.backends.cudnn.benchmark = True
from torch.distributions import Normal, Categorical
import numpy as np
import os
import copy
from collections import OrderedDict

from core.network import Network
from core.optimizer import Optimizer
from .utils import Rollout
from .base import BaseAgent

class REINFORCEAgent(BaseAgent):
    def __init__(self,
                 state_size,
                 action_size,
                 network="discrete_policy",
                 optim_config={'name':'adam'},
                 gamma=0.99,
                 device=None,
                 ):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_type = network.split("_")[0]
        assert self.action_type in ["continuous", "discrete"]

        self.network = Network(network, state_size, action_size).to(self.device)
        self.optimizer = Optimizer(**optim_config, params=self.network.parameters())

        self.gamma = gamma
        self.memory = Rollout()

    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training)
        
        if self.action_type == "continuous":
            mu, std = self.network(torch.as_tensor(state, dtype=torch.float32, device=self.device))
            z = torch.normal(mu, std) if training else mu
            action = torch.tanh(z)
            action = action.cpu().numpy()
        else:
            pi = self.network(torch.as_tensor(state, dtype=torch.float32, device=self.device))
            action = torch.multinomial(pi, 1) if training else torch.argmax(pi, dim=-1, keepdim=True)
        return action.cpu().numpy()
    
    def learn(self):
        state, action, reward = self.memory.rollout()[:3]
        
        ret = np.copy(reward)
        for t in reversed(range(len(ret)-1)):
            ret[t] += self.gamma * ret[t+1]
        
        state, action, ret = map(lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device), [state, action, ret])
        
        if self.action_type == "continuous":
            mu, std = self.network(state)
            m = Normal(mu, std)
            z = torch.atanh(torch.clamp(action, -1+1e-7, 1-1e-7))
            log_prob = m.log_prob(z)
            log_prob -= torch.log(1 - action.pow(2) + 1e-7)
            log_prob = log_prob.sum(1, keepdim=True)
            loss = -(log_prob*ret).mean()
        else:
            pi = self.network(state)
            loss = -(torch.log(pi.gather(1, action.long()))*ret).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        result = {
            'loss' : loss.item()
        }
        return result

    def process(self, transitions, step):
        result = {}
        # Process per step
        self.memory.store(transitions)

        # Process per epi
        if transitions[-1] :
            result = self.learn()
        
        return result

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
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        