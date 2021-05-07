import torch
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
                 optimizer="adam",
                 learning_rate=1e-4,
                 gamma=0.99,
                 **kwargs,
                 ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_type = network.split("_")[0]
        assert self.action_type in ["continuous", "discrete"]

        self.network = Network(network, state_size, action_size).to(self.device)
        self.optimizer = Optimizer(optimizer, self.network.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.memory = Rollout()

    def act(self, state, training=True):
        if self.action_type == "continuous":
            mu, std = self.network(torch.FloatTensor(state).to(self.device))
            std = std if training else 0
            m = Normal(mu, std)
            z = m.sample()
            action = torch.tanh(z)
            action = action.data.cpu().numpy()
        else:
            pi = self.network(torch.FloatTensor(state).to(self.device))
            m = Categorical(pi)
            action = m.sample().data.cpu().numpy()[..., np.newaxis]
        return action

    def learn(self):
        state, action, reward = self.memory.rollout()[:3]
        
        ret = np.copy(reward)
        for t in reversed(range(len(ret)-1)):
            ret[t] += self.gamma * ret[t+1]
        
        state, action, ret = map(lambda x: torch.FloatTensor(x).to(self.device), [state, action, ret])
        
        if self.action_type == "continuous":
            mu, std = self.network(state)
            m = Normal(mu, std)
            z = torch.atanh(torch.clamp(action, -1+1e-7, 1-1e-7))
            log_prob = m.log_prob(z)
            log_prob -= torch.log(1 - action.pow(2) + 1e-7)
            loss = -(log_prob*ret).mean()
        else:
            pi = self.network(state)
            loss = -(torch.log(pi.gather(1, action.long()))*ret).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        result = {
            'loss' : loss.item()
        }
        return result

    def process(self, transitions):
        result = None
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
        
    def cpu(self):
        clone = copy.deepcopy(self)
        clone.device = torch.device("cpu")
        clone.network.cpu()
        clone.optimizer.state.clear()
        clone.memory.clear()
        return clone
    
    def sync_out(self, device="cpu"):
        weights = OrderedDict([(k, v.to(device)) for k, v in self.network.state_dict().items()])
        sync_item ={
            "weights": weights,
        }
        return sync_item
    
    def sync_in(self, weights):
        self.network.load_state_dict(weights)
    
    def set_distributed(self, id):
        return self