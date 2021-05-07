import torch
import torch.nn.functional as F
import numpy as np
import os
import copy
import time

from core.network import Network
from core.optimizer import Optimizer
from .utils import ReplayBuffer
from .dqn import DQNAgent

class NoisyAgent(DQNAgent):
    def __init__(self,
                state_size,
                action_size,
                network='noisy',
                optimizer='adam',
                learning_rate=3e-4,
                opt_eps=1e-8,
                gamma=0.99,
                explore_step=90000,
                buffer_size=50000,
                batch_size=64,
                start_train_step=2000,
                target_update_term=500,
                ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.network = Network(network, state_size, action_size, self.device).to(self.device)
        self.target_network = copy.deepcopy(self.network)
        self.optimizer = Optimizer(optimizer, self.network.parameters(), lr=learning_rate, eps=opt_eps)
        self.gamma = gamma
        self.explore_step = explore_step
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.start_train_step = start_train_step
        self.target_update_term = target_update_term
        self.num_learn = 0
        
    def act(self, state, training=True):
        self.network.train(training)
        
        if training and self.memory.size < max(self.batch_size, self.start_train_step):
            action = np.random.randint(0, self.action_size, size=(state.shape[0], 1))
        else:
            action = torch.argmax(self.network(torch.FloatTensor(state).to(self.device), training), -1, keepdim=True).data.cpu().numpy()
        return action

    def learn(self):
        if self.memory.size < max(self.batch_size, self.start_train_step):
            return None

        transitions = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(self.device), transitions)
        
        eye = torch.eye(self.action_size).to(self.device)
        one_hot_action = eye[action.view(-1).long()]
        q = (self.network(state, True) * one_hot_action).sum(1, keepdims=True)

        with torch.no_grad():
            max_Q = torch.max(q).item()
            next_q = self.target_network(next_state, False)
            target_q = reward + (1 - done) * self.gamma * next_q.max(1, keepdims=True).values
        
        loss = F.smooth_l1_loss(q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.num_learn += 1

        result = {
            "loss" : loss.item(),
            "max_Q": max_Q,
        }

        return result
    
    def process(self, state, action, reward, next_state, done):
        result = None
        # Process per step
        self.memory.store(state, action, reward, next_state, done)
        result = self.learn()

        # Process per step if train start
        if self.num_learn > 0:
            if self.num_learn % self.target_update_term == 0:
                self.update_target()
        
        # Process per episode
        if done.all():
            pass
    
        return result