import torch
import torch.nn.functional as F
import numpy as np
import os
import copy
import time

from core.network import Network
from core.optimizer import Optimizer
from .utils import ReplayBuffer
from .qrdqn import QRDQNAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class IQNAgent(QRDQNAgent):
    def __init__(self,
                state_size,
                action_size,
                network='iqn',
                optimizer='adam',
                learning_rate=3e-4,
                opt_eps=1e-8,
                gamma=0.99,
                epsilon_init=1.0,
                epsilon_min=0.1,
                epsilon_eval=0.0,
                explore_step=90000,
                buffer_size=50000,
                batch_size=64,
                start_train_step=2000,
                target_update_term=500,
                num_sample=64,
                embedding_dim=64,
                sample_min=0.0,
                sample_max=1.0
                ):
        
        self.action_size = action_size
        self.network = Network(network, state_size, action_size, embedding_dim, num_sample).to(device)
        self.target_network = copy.deepcopy(self.network)
        self.optimizer = Optimizer(optimizer, self.network.parameters(), lr=learning_rate, eps=opt_eps)
        self.gamma = gamma
        self.epsilon = epsilon_init
        self.epsilon_init = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_eval = epsilon_eval
        self.explore_step = explore_step
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.start_train_step = start_train_step
        self.target_update_term = target_update_term
        self.num_learn = 0
        
        self.num_support = num_sample
        self.embedding_dim = embedding_dim
        self.sample_min = sample_min
        self.sample_max = sample_max
        
        # Get tau
        min_tau = 1/(2*self.num_support)
        max_tau = (2*self.num_support+1)/(2*self.num_support)
        self.tau = torch.arange(min_tau, max_tau, 1/self.num_support, device=device).view(1, self.num_support)
        self.inv_tau = 1 - self.tau


    def act(self, state, training=True):
        self.network.train(training)
        epsilon = self.epsilon if training else self.epsilon_eval
        sample_min = 0 if training else self.sample_min
        sample_max = 0 if training else self.sample_max
        
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.action_size, size=(state.shape[0], 1))
        else:
            logits = self.network(torch.FloatTensor(state).to(device), sample_min, sample_max)
            _, q_action = self.logits2Q(logits)
            action = torch.argmax(q_action, -1, keepdim=True).data.cpu().numpy()
        return action
    
    def logits2Q(self, logits):
        _logits = torch.transpose(logits, 1, 2)

        q_action = torch.mean(_logits, dim=-1)
        return _logits, q_action
    
