import torch
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
import numpy as np

from core.network import Network
from core.optimizer import Optimizer
from .dqn import DQN

class Noisy(DQN):
    def __init__(self, 
                 state_size,
                 action_size,
                 network='noisy',
                 head='mlp',
                 optim_config={'name':'adam'},
                 # Noisy
                 noise_type='factorized', 
                 **kwargs):
        super(Noisy, self).__init__(state_size, action_size, network=network, head=head, **kwargs)
    
        self.network = Network(network, state_size, action_size, noise_type, head=head).to(self.device)
        self.target_network = Network(network, state_size, action_size, noise_type, head=head).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = Optimizer(**optim_config, params=self.network.parameters())
        
    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training)
        
        if training and self.memory.size < max(self.batch_size, self.start_train_step):
            action = np.random.randint(0, self.action_size, size=(state.shape[0], 1))
        else:
            action = torch.argmax(self.network(torch.as_tensor(state, dtype=torch.float32, device=self.device), training), -1, keepdim=True).cpu().numpy()
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
        
        eye = torch.eye(self.action_size).to(self.device)
        one_hot_action = eye[action.view(-1).long()]
        q = (self.network(state, True) * one_hot_action).sum(1, keepdims=True)

        with torch.no_grad():
            max_Q = torch.max(q).item()
            next_q = self.target_network(next_state, True)
            target_q = reward + (1 - done) * self.gamma * next_q.max(1, keepdims=True).values
        
        loss = F.smooth_l1_loss(q, target_q)
        
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        
        self.num_learn += 1
        
        sig1_mean, sig2_mean = self.network.get_sig_w_mean()
        
        result = {
            "loss" : loss.item(),
            "max_Q": max_Q,
            "sig_w1": sig1_mean.item(),
            "sig_w2": sig2_mean.item(),
        }

        return result
    
    def process(self, transitions, step):
        result = {}
        
        # Process per step
        self.memory.store(transitions)
        delta_t = step - self.time_t
        self.time_t = step
        self.target_update_stamp += delta_t
        
        if self.memory.size >= self.batch_size and self.time_t >= self.start_train_step:
            result = self.learn()

        # Process per step if train start
        if self.num_learn > 0:
            if self.target_update_stamp >= self.target_update_period:
                self.update_target()
                self.target_update_stamp = 0
        
        return result