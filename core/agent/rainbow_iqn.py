import torch
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
import numpy as np
import copy
import time

from core.network import Network
from core.optimizer import Optimizer
from .utils import RainbowBuffer
from .rainbow import RainbowAgent

class RainbowIQNAgent(RainbowAgent):
    def __init__(self,
                state_size,
                action_size,
                network='rainbow_iqn',
                optimizer='adam',
                learning_rate=3e-4,
                opt_eps=1e-8,
                gamma=0.99,
                explore_step=90000,
                buffer_size=50000,
                batch_size=64,
                start_train_step=2000,
                target_update_period=500,
                # MultiStep
                n_step = 4,
                # PER
                alpha = 0.6,
                beta = 0.4,
                learn_period = 4,
                uniform_sample_prob = 1e-3,
                # IQN
                num_sample = 64,
                embedding_dim = 64,
                sample_min = 0.0,
                sample_max = 1.0,
                device = None,
                ):
        
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size        
        self.network = Network(network, state_size, action_size, embedding_dim, num_sample, self.device).to(self.device)
        self.target_network = copy.deepcopy(self.network)
        self.optimizer = Optimizer(optimizer, self.network.parameters(), lr=learning_rate, eps=opt_eps)
        self.gamma = gamma
        self.explore_step = explore_step
        self.batch_size = batch_size
        self.start_train_step = start_train_step
        self.target_update_stamp = 0
        self.target_update_period = target_update_period
        self.num_learn = 0
        self.time_t = 0
        
        # MultiStep
        self.n_step = n_step
        
        # PER
        self.alpha = alpha
        self.beta = beta
        self.learn_period = learn_period
        self.learn_period_stamp = 0 
        self.uniform_sample_prob = uniform_sample_prob
        self.beta_add = 1/self.explore_step
        
        # IQN
        self.num_sample = num_sample
        self.embedding_dim = embedding_dim
        self.sample_min = sample_min
        self.sample_max = sample_max
        
        # MultiStep
        self.memory = RainbowBuffer(buffer_size, self.n_step, self.uniform_sample_prob)
        
    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training)
        sample_min = 0 if training else self.sample_min
        sample_max = 1 if training else self.sample_max
        
        if training and self.memory.size < max(self.batch_size, self.start_train_step):
            action = np.random.randint(0, self.action_size, size=(state.shape[0], 1))
        else:
            logits, _ = self.network(torch.FloatTensor(state).to(self.device), training, sample_min, sample_max)
            _, q_action = self.logits2Q(logits)
            action = torch.argmax(q_action, -1, keepdim=True).cpu().numpy()
        return action

    def learn(self):
        transitions, weights, indices, sampled_p, mean_p = self.memory.sample(self.beta, self.batch_size)
        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(self.device), transitions)
        
         # Get Theta Pred, Tau
        logit, tau = self.network(state, True)
        logits, q_action = self.logits2Q(logit)
        action_eye = torch.eye(self.action_size, device=self.device)
        action_onehot = action_eye[action[:,0].long()]

        theta_pred = action_onehot @ logits
        tau = torch.transpose(tau, 1, 2)
        
        with torch.no_grad():
            # Get Theta Target 
            logit_next, _ = self.network(next_state, False)
            _, q_next = self.logits2Q(logit_next)

            logit_target, _ = self.target_network(next_state, False)
            logits_target, _ = self.logits2Q(logit_target)
            
            max_a = torch.argmax(q_next, axis=-1, keepdim=True)
            max_a_onehot = action_eye[max_a.long()]
            
            theta_target = torch.squeeze(max_a_onehot @ logits_target, 1)
            for i in reversed(range(self.n_step)):
                theta_target = reward[:,i] + (1-done[:,i]) * self.gamma * theta_target
            theta_target = torch.unsqueeze(theta_target, 2)
        
        error_loss = theta_target - theta_pred 
        huber_loss = F.smooth_l1_loss(theta_target, theta_pred, reduction='none')
        
        # Get Loss
        loss = torch.where(error_loss < 0.0, 1-tau, tau) * huber_loss
        loss = torch.mean(torch.sum(loss, axis = 2), axis=1)

        max_Q = torch.max(q_action).item()
        max_logit = torch.max(logit).item()
        min_logit = torch.min(logit).item()
        
        # PER
        p_j = torch.pow(loss, self.alpha)
        
        for i, p in zip(indices, p_j):
            self.memory.update_priority(p.item(), i)
        
        # Annealing beta
        self.beta = min(1.0, self.beta + self.beta_add)

        weights = torch.unsqueeze(torch.FloatTensor(weights).to(self.device), -1)
                    
        loss = (weights * loss).mean()
                
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        
        self.num_learn += 1

        result = {
            "loss" : loss.item(),
            "max_Q": max_Q,
            "max_logit": max_logit,
            "min_logit": min_logit,
            "sampled_p": sampled_p,
            "mean_p": mean_p,
        }

        return result
    
    def logits2Q(self, logits):
        _logits = torch.transpose(logits, 1, 2)

        q_action = torch.mean(_logits, dim=-1)
        return _logits, q_action