import torch
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
import numpy as np
import copy

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
                target_update_period=500,
                 device=None,
                 ):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.network = Network(network, state_size, action_size, self.device).to(self.device)
        self.target_network = copy.deepcopy(self.network)
        self.optimizer = Optimizer(optimizer, self.network.parameters(), lr=learning_rate, eps=opt_eps)
        self.gamma = gamma
        self.explore_step = explore_step
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.start_train_step = start_train_step
        self.target_update_period = target_update_period
        self.target_update_stamp = 0
        self.num_learn = 0
        self.time_t = 0
        
    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training)
        
        if training and self.memory.size < max(self.batch_size, self.start_train_step):
            action = np.random.randint(0, self.action_size, size=(state.shape[0], 1))
        else:
            action = torch.argmax(self.network(torch.as_tensor(state, dtype=torch.float32, device=self.device), training), -1, keepdim=True).cpu().numpy()
        return action

    def learn(self):
        transitions = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = map(lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device), transitions)
        
        eye = torch.eye(self.action_size).to(self.device)
        one_hot_action = eye[action.view(-1).long()]
        q = (self.network(state, True) * one_hot_action).sum(1, keepdims=True)

        with torch.no_grad():
            max_Q = torch.max(q).item()
            next_q = self.target_network(next_state, False)
            target_q = reward + (1 - done) * self.gamma * next_q.max(1, keepdims=True).values
        
        loss = F.smooth_l1_loss(q, target_q)
        
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        
        self.num_learn += 1

        result = {
            "loss" : loss.item(),
            "max_Q": max_Q,
        }

        return result
    
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
            if self.target_update_stamp > self.target_update_period:
                self.update_target()
                self.target_update_stamp = 0
        
        return result