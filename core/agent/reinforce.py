import torch
from torch.distributions import Normal
import numpy as np
import os

from core.network import Network
from core.optimizer import Optimizer
from .utils import ReplayBuffer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class REINFORCEAgent:
    def __init__(self,
                 state_size,
                 action_size,
                 network="discrete_policy",
                 optimizer="adam",
                 learning_rate=1e-4,
                 gamma=0.99,
                 **kwargs,
                 ):
        
        self.action_type = network.split("_")[0]
        assert self.action_type in ["continuous", "discrete"]

        self.network = Network(network, state_size, action_size).to(device)
        self.optimizer = Optimizer(optimizer, self.network.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.memory = ReplayBuffer()

    def act(self, state, training=True):
        if self.action_type == "continuous":
            mu, std = self.network(torch.FloatTensor(state).to(device))
            std = std if training else 0
            m = Normal(mu, std)
            action = m.sample().data.cpu().numpy()
        else:
            pi = self.network(torch.FloatTensor(state).to(device))
            pi = pi.data.cpu().numpy()
            random_choice = lambda x: np.random.choice(pi.shape[-1], p=x, size=(1))
            action = np.array(list(map(random_choice, pi))) if training \
                        else np.argmax(pi, axis= -1)[..., np.newaxis]
        return action

    def learn(self):
        state, action, reward = self.memory.rollout()[:3]

        for t in reversed(range(len(reward)-1)):
            reward[t] += self.gamma * reward[t+1]
        
        state, action, reward = map(lambda x: torch.FloatTensor(x).to(device), [state, action, reward])
        
        if self.action_type == "continuous":
            mu, std = self.network(state)
            m = Normal(mu, std)
            loss = -(m.log_prob(action)*reward).mean()
        else:
            pi = self.network(state)
            loss = -(torch.log(pi.gather(1, action.long()))*reward).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        result = {
            'loss' : loss.item()
        }
        return result

    def process(self, state, action, reward, next_state, done):
        result = None
        # Process per step
        self.memory.store(state, action, reward, next_state, done)

        # Process per epi
        if done :
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
        checkpoint = torch.load(os.path.join(path,"ckpt"),map_location=device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])