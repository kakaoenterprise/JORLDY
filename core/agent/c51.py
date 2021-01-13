import torch
import torch.nn.functional as F
import random
import os
import numpy as np 

from .dqn import DQNAgent
from core.network import Network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class C51Agent(DQNAgent):
    def __init__(self, state_size, action_size, v_min, v_max, num_support , **kwargs):
        super(C51Agent, self).__init__(state_size, action_size*num_support, **kwargs)  
        self.action_size = action_size 
        
        self.v_min = v_min
        self.v_max = v_max
        self.num_support = num_support 
        self.delta_z = (self.v_max - self.v_min) / (self.num_support - 1)
        self.z = torch.linspace(self.v_min, self.v_max, self.num_support, device=device).view(1, -1)
        
    def act(self, state, training=True):
        if random.random() < self.epsilon and training:
            self.network.train()
            action = random.randint(0, self.action_size-1)
        else:
            self.network.eval()
            logits = self.network(torch.FloatTensor(state).to(device))
            _, q_action = self.logits2Q(logits)
            action = torch.argmax(q_action).item()
        return action
    
    def learn(self):        
        if self.memory.length < max(self.batch_size, self.start_train_step):
            return None

        transitions = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device), transitions)
        
        p_logit, q_action = self.logits2Q(self.network(state))
        
        action_eye = torch.eye(self.action_size, device=device)
        action_onehot = action_eye[action.view(-1).long()]
        action_binary = torch.unsqueeze(action_onehot, -1).repeat(1,1,self.num_support)
        p_action = torch.sum(action_binary * p_logit, 1) 
        
        target_dist = torch.zeros(self.batch_size, self.num_support, device=device, requires_grad=False)
    
        with torch.no_grad():
            target_p_logit, target_q_action = self.logits2Q(self.target_network(next_state))
            
            target_action = torch.argmax(target_q_action, -1, keepdims=True)
            target_action_onehot = action_eye[target_action.view(-1).long()]
            target_action_binary = torch.unsqueeze(target_action_onehot, -1).repeat(1,1,self.num_support)
            target_p_action = torch.sum(target_action_binary * target_p_logit, 1)
            
            Tz = reward.expand(-1,self.num_support) + (1-done)*self.gamma*self.z
            
            b = torch.clamp(Tz - self.v_min, 0, self.v_max - self.v_min)/ self.delta_z
            l = torch.floor(b).long()
            u = torch.ceil(b).long()
            
            support_eye = torch.eye(self.num_support, device=device)
            l_support_onehot = support_eye[l]
            u_support_onehot = support_eye[u]
            
            l_support_binary = torch.unsqueeze(u-b, -1).repeat(1,1,self.num_support)
            u_support_binary = torch.unsqueeze(b-l, -1).repeat(1,1,self.num_support)
            
            target_dist = torch.sum(l_support_onehot * l_support_binary + u_support_onehot * u_support_binary, 1)
            target_dist += done * torch.mean(l_support_onehot * u_support_onehot, 1)
            target_dist += (1 - done)*(target_p_action - 1)*target_dist
            target_dist /= done + (1 - done) * torch.sum(target_dist, 1, keepdim=True)
        
        max_Q = torch.max(q_action).item()

        loss = -(target_dist*p_action.log()).sum(-1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.num_learn += 1
        
        result = {
            "loss" : loss.item(),
            "epsilon" : self.epsilon,
            "max_Q": max_Q,
        }
        return result
    
    def logits2Q(self, logits):
        _logits = logits.view(-1, self.action_size, self.num_support)
        p_logit = F.softmax(_logits, dim=-1)

        z_action = self.z.expand(p_logit.shape[0], self.action_size, self.num_support)
        q_action = torch.sum(z_action * p_logit, dim=-1)
        
        return p_logit, q_action 