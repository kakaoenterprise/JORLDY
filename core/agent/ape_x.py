from collections import deque
import torch
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
import numpy as np
import copy

from core.network import Network
from core.optimizer import Optimizer
from core.buffer import ApeXBuffer
from .dqn import DQN

class ApeX(DQN):
    def __init__(self,
                 # ApeX
                 epsilon = 0.4,
                 epsilon_alpha = 0.7,                 
                 clip_grad_norm = 40.0,
                 n_epoch = 16,
                 # PER
                 alpha = 0.6,
                 beta = 0.4,
                 learn_period = 4,
                 uniform_sample_prob = 1e-3,
                 # MultiStep
                 n_step = 4,
                 **kwargs
                 ):
        super(ApeX, self).__init__(**kwargs)
        # ApeX
        self.epsilon = epsilon
        self.epsilon_alpha = epsilon_alpha
        self.clip_grad_norm = clip_grad_norm
        self.transition_buffer = deque(maxlen=n_step)
        self.time_t = n_step - 1 # for sync between step and # of transitions
        self.n_epoch = n_epoch
        
        # PER
        self.alpha = alpha
        self.beta = beta
        self.learn_period = learn_period
        self.learn_period_stamp = 0 
        self.uniform_sample_prob = uniform_sample_prob
        self.beta_add = 1/self.explore_step
        
        # MultiStep
        self.n_step = n_step
        self.memory = ApeXBuffer(self.gamma, self.buffer_size, self.n_step, self.uniform_sample_prob)
    
    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training)
        epsilon = self.epsilon if training else self.epsilon_eval
            
        q = self.network(torch.as_tensor(state, dtype=torch.float32, device=self.device))
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.action_size, size=(state.shape[0], 1))
        else:
            action = torch.argmax(q, -1, keepdim=True).cpu().numpy()
        q = np.take(q.cpu().numpy(), action)
        return {'action': action, 'q': q}
    
    def learn(self):
        losses, max_Qs, sampled_ps, mean_ps = [], [], [], []
        for _ in range(self.n_epoch):
            transitions, weights, indices, sampled_p, mean_p = self.memory.sample(self.beta, self.batch_size)
            for key in transitions.keys():
                transitions[key] = torch.as_tensor(transitions[key], dtype=torch.float32, device=self.device)

            state = transitions['state']
            action = transitions['action']
            reward = transitions['reward']
            next_state = transitions['next_state']
            done = transitions['done']

            eye = torch.eye(self.action_size).to(self.device)
            one_hot_action = eye[action[:, 0].view(-1).long()]
            q = (self.network(state) * one_hot_action).sum(1, keepdims=True)

            with torch.no_grad():
                max_Q = torch.max(q).item()
                next_q = self.network(next_state)
                max_a = torch.argmax(next_q, axis=1)
                max_eye = torch.eye(self.action_size).to(self.device)
                max_one_hot_action = eye[max_a.view(-1).long()]

                next_target_q = self.target_network(next_state)
                target_q = (next_target_q * max_one_hot_action).sum(1, keepdims=True)

                for i in reversed(range(self.n_step)):
                    target_q = reward[:, i] + (1 - done[:, i]) * self.gamma * target_q

            # Update sum tree
            td_error = abs(target_q - q)
            p_j = torch.pow(td_error, self.alpha)
            for i, p in zip(indices, p_j):
                self.memory.update_priority(p.item(), i)

            # Annealing beta
            self.beta = min(1.0, self.beta + self.beta_add)

            weights = torch.unsqueeze(torch.FloatTensor(weights).to(self.device), -1)

            loss = (weights * (td_error**2)).mean()        
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            
            losses.append(loss.item())
            max_Qs.append(max_Q)
            sampled_ps.append(sampled_p)
            mean_ps.append(mean_p)
            
        self.num_learn += 1

        result = {
            "loss" : np.mean(losses),
            "max_Q": np.mean(max_Qs),
            "sampled_p": np.mean(sampled_ps),
            "mean_p": np.mean(mean_ps),
        }

        return result
    
    def process(self, transitions, step):
        result = {}
        
        # Process per step
        delta_t = step - self.time_t
        self.memory.store(transitions, delta_t)
        self.time_t = step
        self.target_update_stamp += delta_t
        self.learn_period_stamp += delta_t
        
        if (self.learn_period_stamp > self.learn_period and
            self.memory.buffer_counter > self.batch_size and
            self.time_t >= self.start_train_step):
            result = self.learn()
            self.learn_period_stamp = 0

        # Process per step if train start
        if self.num_learn > 0 and self.target_update_stamp > self.target_update_period:
            self.update_target()
            self.target_update_stamp = 0
            
        return result

    def set_distributed(self, id, num_worker):
        self.epsilon = self.epsilon**(1 + (id/(num_worker-1))*self.epsilon_alpha)
        return self
    
    def interact_callback(self, transitions):
        _transitions = []
        for transition in transitions:
            self.transition_buffer.append(transition)
            if len(self.transition_buffer) == self.n_step:
                target_q = self.transition_buffer[-1]['q']
                for i in reversed(range(self.n_step)):
                    target_q = self.transition_buffer[i]['reward'] \
                                + (1 - self.transition_buffer[i]['done']) * self.gamma * target_q
                priority = abs(target_q - self.transition_buffer[0]['q'])

                _transition = self.transition_buffer[0].copy()
                _transition['priority'] = priority
                del _transition['q']
                _transitions.append(_transition)
                
        return _transitions
        