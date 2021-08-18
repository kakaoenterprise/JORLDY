import torch
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
import numpy as np
import os
import copy
from collections import OrderedDict

from core.network import Network
from core.optimizer import Optimizer
from core.buffer import ReplayBuffer
from .base import BaseAgent

class DQN(BaseAgent):
    def __init__(self,
                state_size,
                action_size,
                optim_config={'name':'adam'},
                network='dqn',
                gamma=0.99,
                epsilon_init=1.0,
                epsilon_min=0.1,
                epsilon_eval=0.0,
                explore_step=90000,
                buffer_size=50000,
                batch_size=64,
                start_train_step=2000,
                target_update_period=500,
                device=None,
                ):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.network = Network(network, state_size, action_size).to(self.device)
        self.target_network = copy.deepcopy(self.network)
        self.optimizer = Optimizer(**optim_config, params=self.network.parameters())
        self.gamma = gamma
        self.epsilon = epsilon_init
        self.epsilon_init = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_eval = epsilon_eval
        self.explore_step = explore_step
        self.epsilon_delta = (epsilon_init - epsilon_min)/explore_step
        self.buffer_size = buffer_size
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.start_train_step = start_train_step
        self.target_update_stamp = 0
        self.target_update_period = target_update_period
        self.num_learn = 0
        self.time_t = 0
    
    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training)
        epsilon = self.epsilon if training else self.epsilon_eval
            
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.action_size, size=(state.shape[0], 1))
        else:
            action = torch.argmax(self.network(torch.as_tensor(state, dtype=torch.float32, device=self.device)), -1, keepdim=True).cpu().numpy()
        return {'action': action}

    def learn(self):
        transitions = self.memory.sample(self.batch_size)
        for key in transitions.keys():
            transitions[key] = torch.as_tensor(transitions[key], dtype=torch.float32, device=self.device)

        state = transitions['state']
        action = transitions['action']
        reward = transitions['reward']
        next_state = transitions['next_state']
        done = transitions['done']
        
        eye = torch.eye(self.action_size, device=self.device)
        one_hot_action = eye[action.view(-1).long()]
        q = (self.network(state) * one_hot_action).sum(1, keepdims=True)
        with torch.no_grad():
            max_Q = torch.max(q).item()
            next_q = self.target_network(next_state)
            target_q = reward + (1 - done) * self.gamma * next_q.max(1, keepdims=True).values

        loss = F.smooth_l1_loss(q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.num_learn += 1

        result = {
            "loss" : loss.item(),
            "epsilon" : self.epsilon,
            "max_Q": max_Q,
        }
        
        return result

    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())
        
    def process(self, transitions, step):
        result = {}

        # Process per step
        self.memory.store(transitions)
        delta_t = step - self.time_t
        self.time_t = step
        self.target_update_stamp += delta_t
        
        if self.memory.size > self.batch_size and self.time_t >= self.start_train_step:
            result = self.learn()

        # Process per step if train start
        if self.num_learn > 0:
            self.epsilon_decay(delta_t)

            if self.target_update_stamp > self.target_update_period:
                self.update_target()
                self.target_update_stamp = 0

        return result
            
    def epsilon_decay(self, delta_t):
        new_epsilon = self.epsilon - delta_t*self.epsilon_delta
        self.epsilon = max(self.epsilon_min, new_epsilon)

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
        self.target_network = copy.deepcopy(self.network)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
    
    def set_distributed(self, id, num_worker):
        self.epsilon = id / num_worker
        return self