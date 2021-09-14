from collections import deque
import torch
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
import numpy as np
import copy

from core.network import Network
from core.optimizer import Optimizer
from core.buffer import PERBuffer
from .dqn import DQN

import time

class R2D2(DQN):
    def __init__(self,
                 state_size,
                 action_size,
                 # ApeX
                 epsilon = 0.4,
                 epsilon_alpha = 0.7,                 
                 clip_grad_norm = 40.0,
                 # PER
                 alpha = 0.6,
                 beta = 0.4,
                 learn_period = 4,
                 uniform_sample_prob = 1e-3,
                 # MultiStep
                 n_step = 4,
                 # R2D2
                 seq_len = 80,
                 n_burn_in = 20,
                 **kwargs
                 ):
        super(R2D2, self).__init__(state_size=state_size, action_size=action_size, **kwargs)
        self.state_size = state_size 
        
        # ApeX
        self.epsilon = epsilon
        self.epsilon_alpha = epsilon_alpha
        self.clip_grad_norm = clip_grad_norm
        self.tmp_buffer = deque(maxlen=n_step)
        self.time_t = n_step - 1 # for sync between step and # of transitions
        
        # PER
        self.alpha = alpha
        self.beta = beta
        self.learn_period = learn_period
        self.learn_period_stamp = 0 
        self.uniform_sample_prob = uniform_sample_prob
        self.beta_add = 1/self.explore_step
        
        # MultiStep
        self.n_step = n_step
        
        # R2D2
        self.seq_len = seq_len
        self.n_burn_in = n_burn_in
        
        self.memory = PERBuffer(self.buffer_size, uniform_sample_prob)
        self.state_seq = None 
        self.state_list = []
        self.hidden_list = []
    
    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training)
        epsilon = self.epsilon if training else self.epsilon_eval
        
        if self.state_seq is None:
            self.state_seq = np.repeat(np.zeros_like(state), self.seq_len, axis=0)
            self.state_list = []
            self.hidden_list = []
        
        init_state_seq = self.state_seq[0]
        self.state_list.append(state)
        
        if np.all(init_state_seq) == np.all(self.state_list[0][0]):
            hidden_h = self.hidden_list[0][0]
            hidden_c = self.hidden_list[0][1]
            
            del self.hidden_list[0]
            del self.state_list[0]
        else:
            hidden_h = torch.zeros(1,1,512)
            hidden_c = torch.zeros(1,1,512)
        
        init_hidden = (hidden_h.contiguous(), hidden_c.contiguous())
        
        self.state_seq = np.concatenate([self.state_seq[1:], state], axis=0)
        
        q, hidden = self.network(torch.as_tensor([self.state_seq], dtype=torch.float32, device=self.device), self.seq_len, self.n_burn_in, init_hidden)
        
        self.hidden_list.append(hidden)

        if np.random.random() < epsilon:
            action = np.random.randint(0, self.action_size, size=(state.shape[0], 1))
        else:
            action = torch.argmax(q, -1, keepdim=True).cpu().numpy()
        q = np.take(q.cpu().numpy(), action)
        
        hidden_h = hidden_h.cpu().numpy()
        hidden_c = hidden_c.cpu().numpy()
        
        return {'action': action, 'q': q, 'state_seq': self.state_seq, 'hidden_h': hidden_h, 'hidden_c': hidden_c}
    
    def learn(self):
        transitions, weights, indices, sampled_p, mean_p = self.memory.sample(self.beta, self.batch_size)
        for key in transitions.keys():
            transitions[key] = torch.as_tensor(transitions[key], dtype=torch.float32, device=self.device)

        state = transitions['state']
        action = transitions['action']
        reward = transitions['reward']
        next_state = transitions['next_state']
        done = transitions['done']
        state_seq = transitions['state_seq']
        nstep_seq = transitions['nstep_seq']
        hidden_h = transitions['hidden_h'].transpose(0,2)
        hidden_c = transitions['hidden_c'].transpose(0,2)
                        
        hidden = (hidden_h[:,0].contiguous(), hidden_c[:,0].contiguous())
        next_hidden = (hidden_h[:,-1].contiguous(), hidden_c[:,-1].contiguous())
        
        # Next state seq 
        next_state_seq = torch.cat((state_seq[:,self.n_step-self.seq_len:], nstep_seq), axis=1)

        eye = torch.eye(self.action_size).to(self.device)
        one_hot_action = eye[action[:, 0].view(-1).long()]
        q_pred, _ = self.network(state_seq, self.seq_len, self.n_burn_in, hidden)
        q = (q_pred * one_hot_action).sum(1, keepdims=True)

        with torch.no_grad():
            max_Q = torch.max(q).item()
            next_q, _ = self.network(next_state_seq, self.seq_len, self.n_burn_in, next_hidden)
            max_a = torch.argmax(next_q, axis=1)
            max_eye = torch.eye(self.action_size).to(self.device)
            max_one_hot_action = eye[max_a.view(-1).long()]

            next_target_q, _ = self.target_network(next_state_seq, self.seq_len, self.n_burn_in, next_hidden)
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

        self.num_learn += 1

        result = {
            "loss" : loss.item(),
            "max_Q": max_Q,
            "sampled_p": sampled_p,
            "mean_p": mean_p,
        }

        return result
    
    def process(self, transitions, step):
        result = {}
                
        # Process per step
        delta_t = step - self.time_t
        if step%(self.seq_len//2)==0:
            self.memory.store(transitions)
                
        self.time_t = step
        self.target_update_stamp += delta_t
        self.learn_period_stamp += delta_t
        
        if (self.learn_period_stamp >= self.learn_period and
            self.memory.buffer_counter >= self.batch_size and
            self.time_t >= self.start_train_step):
            result = self.learn()
            self.learn_period_stamp = 0

        # Process per step if train start
        if self.num_learn > 0 and self.target_update_stamp >= self.target_update_period:
            self.update_target()
            self.target_update_stamp = 0
            
        return result

    def set_distributed(self, id):
        self.epsilon = self.epsilon**(1 + (id/(self.num_workers-1))*self.epsilon_alpha)
        return self
    
    def interact_callback(self, transition):
        _transition = {}
        self.tmp_buffer.append(transition)
        if len(self.tmp_buffer) == self.n_step:
#             _transition['state'] = self.tmp_buffer[0]['state']
#             _transition['next_state'] = self.tmp_buffer[-1]['next_state']
            _transition['state_seq'] = self.tmp_buffer[0]['state_seq'][np.newaxis, ...]

            for key in self.tmp_buffer[0].keys():
#                 if key not in ['state', 'next_state', 'state_seq']:
                if key not in ['state_seq']:
                    _transition[key] = np.stack([t[key] for t in self.tmp_buffer], axis=1)
            
            target_q = self.tmp_buffer[-1]['q']
            for i in reversed(range(self.n_step)):
                target_q = self.tmp_buffer[i]['reward'] \
                            + (1 - self.tmp_buffer[i]['done']) * self.gamma * target_q
            priority = abs(target_q - self.tmp_buffer[0]['q'])
            _transition['priority'] = priority
            del _transition['q']
            
            # Make n-step seq 
#             nstep_array = np.zeros([self.n_step, *self.tmp_buffer[0]['next_state'][0].shape])
#             for i in range(len(self.n_step)):
#                 nstep_array[i] = self.tmp_buffer[i]['next_state']

            nstep_array = np.concatenate([t['next_state'] for t in self.tmp_buffer]) 

            _transition['nstep_seq'] = nstep_array[np.newaxis, ...]
            
        return _transition