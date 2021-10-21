from collections import deque
from itertools import islice
import torch
torch.backends.cudnn.benchmark = True
import numpy as np

from .ape_x import ApeX

class R2D2(ApeX):
    def __init__(self,
                 # R2D2
                 seq_len = 80,
                 n_burn_in = 20,
                 zero_padding = True,
                 eta = 0.9,
                 **kwargs
                 ):
        super(R2D2, self).__init__(**kwargs)
        
        # R2D2
        self.seq_len = seq_len
        self.n_burn_in = n_burn_in
        self.zero_padding = zero_padding
        self.eta = eta
        
        self.hidden = None
        self.tmp_buffer = deque(maxlen=self.n_step + seq_len)
        self.store_period = seq_len//2
        self.store_period_stamp = 0
        self.store_start = True
            
    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training)
        epsilon = self.epsilon if training else self.epsilon_eval
        
        state = np.expand_dims(state, axis=1)
        q, hidden_in, hidden_out = self.network(torch.as_tensor(state, dtype=torch.float32, device=self.device), hidden_in=self.hidden)
        
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.action_size, size=(state.shape[0], 1))
        else:
            action = torch.argmax(q, -1, keepdim=True).cpu().numpy()[:, -1]
        q = np.take(q.cpu().numpy()[:, -1], action)

        hidden_h = hidden_in[0].cpu().numpy()
        hidden_c = hidden_in[1].cpu().numpy()

        self.hidden = hidden_out

        return {'action': action, 'q': q, 'hidden_h': hidden_h, 'hidden_c': hidden_c}
    
    def learn(self):
        transitions, weights, indices, sampled_p, mean_p = self.memory.sample(self.beta, self.batch_size)
        for key in transitions.keys():
            transitions[key] = torch.as_tensor(transitions[key], dtype=torch.float32, device=self.device)

        state = transitions['state'][:,:self.seq_len]
        action = transitions['action']
        reward = transitions['reward'][:,-self.n_step:]
        next_state = transitions['state'][:,self.n_step:]
        done = transitions['done'][:,-self.n_step:]
        hidden_h = transitions['hidden_h'].transpose(0,1).contiguous()
        hidden_c = transitions['hidden_c'].transpose(0,1).contiguous()
        next_hidden_h = transitions['next_hidden_h'].transpose(0,1).contiguous()
        next_hidden_c = transitions['next_hidden_c'].transpose(0,1).contiguous()
        
        hidden = (hidden_h, hidden_c)
        next_hidden = (hidden_h, hidden_c)
        
        eye = torch.eye(self.action_size).to(self.device)
        one_hot_action = eye[action.long()]
        q_pred = self.get_q(state, hidden, self.network)
        q = (q_pred * one_hot_action).sum(-1, keepdims=True)

        with torch.no_grad():
            max_Q = torch.max(q).item()
            next_q = self.get_q(next_state, next_hidden, self.target_network)
            max_a = torch.argmax(next_q, axis=-1)
            max_eye = torch.eye(self.action_size).to(self.device)
            max_one_hot_action = eye[max_a.long()]
                
            next_target_q = self.get_q(next_state, next_hidden, self.target_network)
            target_q = (next_target_q * max_one_hot_action).sum(-1, keepdims=True)
            target_q = self.inv_val_rescale(target_q)
            
            for i in reversed(range(self.n_step)):
                target_q = reward[:, i:i+1] + (1 - done[:, i:i+1]) * self.gamma * target_q
            
            target_q = self.val_rescale(target_q)

        # Update sum tree
        td_error = abs(target_q - q)
        priority = self.eta*torch.max(td_error, axis=1).values + (1-self.eta)*torch.mean(td_error, axis=1)
        p_j = torch.pow(priority, self.alpha)
        for i, p in zip(indices, p_j):
            self.memory.update_priority(p.item(), i)

        # Annealing beta
        self.beta = min(1.0, self.beta + self.beta_add)

        weights = torch.FloatTensor(weights[..., np.newaxis, np.newaxis]).to(self.device)
                
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
    
    def interact_callback(self, transition):
        _transition = {}
        self.tmp_buffer.append(transition)
        
        if (self.store_start or self.store_period_stamp == self.store_period) and\
           ((self.zero_padding and len(self.tmp_buffer) >= self.n_step + 1) or
            (not self.zero_padding and len(self.tmp_buffer) == self.tmp_buffer.maxlen)):
            _transition['action'] = self.tmp_buffer[-self.n_step-1]['action']
            _transition['hidden_h'] = self.tmp_buffer[0]['hidden_h']
            _transition['hidden_c'] = self.tmp_buffer[0]['hidden_c']
            _transition['next_hidden_h'] = self.tmp_buffer[self.n_step]['hidden_h']
            _transition['next_hidden_c'] = self.tmp_buffer[self.n_step]['hidden_c']
            
            for key in self.tmp_buffer[0].keys():
                if key not in ['action', 'hidden_h', 'hidden_c', 'next_state']:
                    if key in ['q', 'state']:
                        _transition[key] = np.stack([t[key] for t in self.tmp_buffer], axis=1)
                    else:
                        # _transition[key] = np.stack([t[key] for t in self.tmp_buffer][-self.n_step-1:-1], axis=1) # if remove zero padding
                        _transition[key] = np.stack([t[key] for t in self.tmp_buffer][:-1], axis=1)
            
            #state sequence zero padding
            if self.zero_padding and len(self.tmp_buffer) < self.tmp_buffer.maxlen:
                lack_dims = self.tmp_buffer.maxlen - len(self.tmp_buffer)
                zero_state = np.zeros((1 , lack_dims, *transition['state'].shape[1:]))
                _transition['state'] = np.concatenate((zero_state, _transition['state']), axis=1)
                zero_reward = np.zeros((1 , lack_dims, *transition['reward'].shape[1:]))
                _transition['reward'] = np.concatenate((zero_reward, _transition['reward']), axis=1)
                zero_done = np.zeros((1 , lack_dims, *transition['done'].shape[1:]))
                _transition['done'] = np.concatenate((zero_done, _transition['done']), axis=1)
                zero_q = np.zeros((1 , lack_dims, *transition['q'].shape[1:]))
                _transition['q'] = np.concatenate((zero_q, _transition['q']), axis=1)
            
            target_q = self.inv_val_rescale(_transition['q'][:, self.n_step:])
            for i in reversed(range(self.n_step)):
                target_q = self.tmp_buffer[-self.n_step-1+i]['reward'] \
                            + (1 - self.tmp_buffer[-self.n_step-1+i]['done']) * self.gamma * target_q
                
            target_q = self.val_rescale(target_q)
            td_error = abs(target_q - _transition['q'][:, :self.seq_len])
            priority = self.eta*np.max(td_error, axis=1) + (1-self.eta)*np.mean(td_error, axis=1)
            _transition['priority'] = priority
            del _transition['q']
            
            self.store_start = False
            self.store_period_stamp = 0
            if self.tmp_buffer[-self.n_step-1]['done']:
                self.tmp_buffer = deque(islice(self.tmp_buffer, len(self.tmp_buffer)-self.n_step, None),
                                        maxlen = self.tmp_buffer.maxlen)
                self.store_start = True

        self.store_period_stamp += 1
        if transition['done']:
            self.hidden = None
        
        return _transition
    
    def get_q(self, state, hidden_in, network):
        with torch.no_grad(): 
            q, hidden_in, hidden_out = network(state[:,:self.n_burn_in], hidden_in)
        q, hidden_in, hidden_out = network(state[:,self.n_burn_in:], hidden_out)
        return q
    
    def val_rescale(self, val, eps=1e-6):
        return (val/(abs(val)+1e-10)) * ((abs(val)+1)**(1/2)-1) + (eps * val)
    
    def inv_val_rescale(self, val, eps=1e-6):
        # Reference: Proposition A.2 in paper "Observe and Look Further: Achieving Consistent Performance on Atari"
        return (val/(abs(val)+1e-10)) * ((((1+4*eps*(abs(val)+1+eps))**(1/2)-1)/(2*eps))**2-1)