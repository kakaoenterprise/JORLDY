import torch
import torch.nn.functional as F
import random

from core.utils import ReplayBuffer

class DQNAgent:
    def __init__(self,
                network,
                target_network,
                optimizer,
                gamma=0.99,
                epsilon_init=0.8,
                epsilon_min=0.001,
                epsilon_decay=0.0001,
                buffer_size=50000,
                batch_size=64,
                start_train=2000,
                update_term=500,
                ):
        self.network = network
        self.target_network = target_network
        self.optimizer = optimizer
        self.gamma = gamma
        self.epsilon = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.start_train = start_train
        self.update_term = update_term
        self.num_learn = 0
        self.action_size = network.D_out
    
    def act(self, state, training=True):
        if random.random() < self.epsilon and training:
            action = random.randint(0, self.action_size-1)
        else:
            action = torch.argmax(self.network(state)).item()
        return action

    def learn(self):
        if self.memory.length < max(self.batch_size, self.start_train):
            return 0

        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        one_hot_action = torch.eye(self.action_size)[action.view(-1).long()]
        q = (self.network(state) * one_hot_action).sum(1, keepdims=True)
        next_q = self.target_network(next_state)
        target_q = reward + next_q.max(1, keepdims=True).values*self.gamma*(1 - done)
        loss = F.smooth_l1_loss(q, target_q).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.num_learn % self.update_term == 0:
            self.update_target()
        self.num_learn += 1

        return loss.item()

    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)
