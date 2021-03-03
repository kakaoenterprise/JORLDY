import torch
import torch.nn.functional as F
import numpy as np
import os

from core.network import Network
from core.optimizer import Optimizer
from .utils import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class IQNAgent:
    def __init__(self,
                state_size,
                action_size,
                network='dqn',
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
        self.target_network = Network(network, state_size, action_size, embedding_dim, num_sample).to(device)
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
        
        self.num_sample = num_sample
        self.embedding_dim = embedding_dim
        self.sample_min = sample_min
        self.sample_max = sample_max        

        self.update_target()

    def act(self, state, training=True):
        self.network.train(training)
        epsilon = self.epsilon if training else self.epsilon_eval
            
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.action_size, size=(state.shape[0], 1))
        else:
            x_embed = self.make_embed(state, self.sample_min, self.sample_max)            
            logits = self.network(torch.FloatTensor(state).to(device), x_embed)
            _, q_action = self.logits2Q(logits)
            action = torch.argmax(q_action, -1, keepdim=True).data.cpu().numpy()
            
        return action
    
    def learn(self):
        if self.memory.size < max(self.batch_size, self.start_train_step):
            return None
        
        transitions = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device), transitions)
        
        # Get Theta Pred
        x_embed = self.make_embed(state, 0, 1)
        logit = self.network(state, x_embed)
        logits, q_action = self.logits2Q(logit)
        
        action_eye = torch.eye(self.action_size, device=device)
        action_onehot = action_eye[action.view(-1).long()]
        action_binary = torch.unsqueeze(action_onehot, -1).repeat(1,1,self.num_sample)
        
        theta_pred = torch.unsqueeze(torch.sum(logits * action_binary, 1), 1).repeat(1,self.num_sample,1)
               
        with torch.no_grad():
            # Get Theta Target 
            x_embed = self.make_embed(next_state, 0, 1)
            logit_next = self.network(next_state, x_embed)
            _, q_next = self.logits2Q(logit_next)
            
            x_embed = self.make_embed(next_state, 0, 1)
            logit_target = self.target_network(next_state, x_embed)
            logits_target, q_target = self.logits2Q(logit_target)
            
            max_a = torch.argmax(q_next, axis=-1)
            max_a_onehot = action_eye[max_a.long()]
            max_a_binary = torch.unsqueeze(max_a_onehot, -1).repeat(1,1,self.num_sample)

            theta_target = reward + torch.sum(logits_target * max_a_binary, 1) * (self.gamma * (1 - done))
            theta_target = torch.unsqueeze(theta_target, -1).repeat(1,1,self.num_sample)
        
        error_loss = theta_target - theta_pred 
        huber_loss = F.smooth_l1_loss(theta_target, theta_pred, reduction='none')

        # Get tau
        min_tau = 1/(2*self.num_sample)
        max_tau = (2*self.num_sample+1)/(2*self.num_sample)
        tau = torch.reshape(torch.arange(min_tau, max_tau, 1/self.num_sample), (1, self.num_sample)).to(device)
        inv_tau = 1.0 - tau

        # Get Loss
        loss = torch.where(error_loss< 0.0, inv_tau * huber_loss, tau * huber_loss)
        loss = torch.mean(torch.sum(torch.mean(loss, axis = 2), axis = 1))                
        
        max_Q = torch.max(q_action).item()
        max_logit = torch.max(logit).item()
        min_logit = torch.min(logit).item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.num_learn += 1
        
        result = {
            "loss" : loss.item(),
            "epsilon" : self.epsilon,
            "max_Q": max_Q,
            "max_logit": max_logit,
            "min_logit": min_logit,
        }
        return result

    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())
        
    def process(self, state, action, reward, next_state, done):
        result = None
        # Process per step
        self.memory.store(state, action, reward, next_state, done)
        result = self.learn()

        # Process per step if train start
        if self.num_learn > 0:
            self.epsilon_decay()

            if self.num_learn % self.target_update_term == 0:
                self.update_target()
        
        # Process per episode
        if done.all():
            pass

        return result
            
    def epsilon_decay(self):
        new_epsilon = self.epsilon - (self.epsilon_init - self.epsilon_min)/(self.explore_step)
        self.epsilon = max(self.epsilon_min, new_epsilon)

    def save(self, path):
        torch.save({
            "network" : self.network.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
        }, os.path.join(path,"ckpt"))

    def load(self, path):
        checkpoint = torch.load(os.path.join(path,"ckpt"),map_location=device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.update_target()

    def logits2Q(self, logits):
        _logits = logits.view(self.num_sample, -1, self.action_size)
        _logits = _logits.transpose(0,1)
        _logits = _logits.transpose(1,2)
        q_action = torch.sum(1/self.num_sample * _logits, dim=-1)
        
        return _logits, q_action
    
    def make_embed(self, x, tau_min, tau_max):
        x_size = x.shape[0]
        
        sample = torch.FloatTensor(x_size*self.num_sample, 1).uniform_(tau_min, tau_max)   
        sample_tile = sample.repeat(1,self.embedding_dim)
        
        embed = torch.cos(torch.arange(0, self.embedding_dim) * 3.141592 * sample_tile)
        
        return embed.to(device)