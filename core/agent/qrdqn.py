import torch
import torch.nn.functional as F
import os
import numpy as np 

from .dqn import DQNAgent
from core.network import Network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QRDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size, num_support , **kwargs):
        super(QRDQNAgent, self).__init__(state_size, action_size*num_support, **kwargs)  

        self.action_size = action_size 
        self.num_support = num_support 
        
    def act(self, state, training=True):
        self.network.train(training)
        epsilon = self.epsilon if training else self.epsilon_eval
            
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.action_size, size=(state.shape[0], 1))
        else:
            logits = self.network(torch.FloatTensor(state).to(device))
            _, q_action = self.logits2Q(logits)
            action = torch.argmax(q_action, -1, keepdim=True).data.cpu().numpy()
        return action
    
    def learn(self):
        if self.memory.size < max(self.batch_size, self.start_train_step):
            return None
        
        transitions = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device), transitions)
        
        # Get Theta Pred
        logit = self.network(state)
        logits, q_action = self.logits2Q(logit)
        
        action_eye = torch.eye(self.action_size, device=device)
        action_onehot = action_eye[action.view(-1).long()]
        action_binary = torch.unsqueeze(action_onehot, -1).repeat(1,1,self.num_support)
        
        theta_pred = torch.unsqueeze(torch.sum(logits * action_binary, 1), 1).repeat(1,self.num_support,1)
        with torch.no_grad():
            # Get Theta Target 
            logit_next = self.network(next_state)
            _, q_next = self.logits2Q(logit_next)

            logit_target = self.target_network(next_state)
            logits_target, q_target = self.logits2Q(logit_target)
            
            max_a = torch.argmax(q_next, axis=-1)
            max_a_onehot = action_eye[max_a.long()]
            max_a_binary = torch.unsqueeze(max_a_onehot, -1).repeat(1,1,self.num_support)
            
            theta_target = reward + torch.sum(logits_target * max_a_binary, 1) * (self.gamma * (1 - done))
            theta_target = torch.unsqueeze(theta_target, 2).repeat(1,1,self.num_support)
        error_loss = theta_target - theta_pred 

        huber_loss = F.smooth_l1_loss(theta_target, theta_pred)

        # Get tau
        min_tau = 1/(2*self.num_support)
        max_tau = (2*self.num_support+1)/(2*self.num_support)
        tau = torch.reshape(torch.arange(min_tau, max_tau, 1/self.num_support), (1, self.num_support)).to(device)
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
    
    def logits2Q(self, logits):
        _logits = logits.view(logits.shape[0], self.action_size, self.num_support)
        q_action = torch.sum(1/self.num_support * _logits, dim=-1)
        
        return _logits, q_action