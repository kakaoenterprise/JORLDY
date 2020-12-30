import torch
import torch.nn.functional as F
import random
import os
import numpy as np 
import time 

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
        self.z = np.reshape(np.linspace(self.v_min, self.v_max, self.num_support), [1,self.num_support])
        
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
        start_time = time.time()
        
        transitions = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device), transitions)
        
        p_logit, q_action = self.logits2Q(self.network(state))
        
        action_binary = np.zeros([self.batch_size, self.action_size, self.num_support])

        for i in range(self.batch_size):
            action_binary[i, int(action[i].item()), :] = 1

        p_current_action = torch.sum(torch.FloatTensor(action_binary).to(device) * p_logit, dim=1) 
        
        z = torch.FloatTensor(self.z)
        
        target_dist = torch.zeros(self.batch_size, self.num_support, requires_grad=False)
        
        with torch.no_grad():
            p_logit_target, q_action_target = self.logits2Q(self.target_network(next_state))
            for i in range(self.batch_size):
                action_max = torch.argmax(q_action_target[i, :]).item()
                reward_ = reward[i].item()
                if done[i]:
                    Tz = reward_

                    # Bounding Tz
                    if Tz >= self.v_max:
                        Tz = self.v_max
                    elif Tz <= self.v_min:
                        Tz = self.v_min

                    b = (Tz - self.v_min) / self.delta_z
                    l = np.int32(np.floor(b))
                    u = np.int32(np.ceil(b))

                    target_dist[i, l] += (u - b)
                    target_dist[i, u] += (b - l)

                    if l==u:
                        target_dist[i,l] = 1
                else:
                    for j in range(self.num_support):
                        Tz = reward_ + self.gamma * self.z[0,j]

                        # Bounding Tz
                        if Tz >= self.v_max:
                            Tz = self.v_max
                        elif Tz <= self.v_min:
                            Tz = self.v_min
                        
                        b = (Tz - self.v_min) / self.delta_z
                        l = np.int32(np.floor(b))
                        u = np.int32(np.ceil(b))

                        target_dist[i, l] += p_logit_target[i, action_max, j].item() * (u - b)
                        target_dist[i, u] += p_logit_target[i, action_max, j].item() * (b - l)

                    sum_target_dist = torch.sum(target_dist[i,:])
                    for j in range(self.num_support):
                        target_dist[i, j] = target_dist[i, j] / sum_target_dist
        
        max_Q = torch.max(q_action).item()

        target_dist_gpu = target_dist.to(device)
        loss = -(target_dist_gpu * p_current_action.log()).sum(-1)
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.num_learn += 1
        
        result = {
            "loss" : loss.item(),
            "epsilon" : self.epsilon,
            "max_Q": max_Q,
        }
#         print(time.time()-start_time)
        return result
    
    def logits2Q(self, logits):
        logits_reshape = torch.reshape(logits, (-1, self.action_size, self.num_support))
        p_logit = F.softmax(logits_reshape, dim=-1)

        z = torch.FloatTensor(self.z)

        z_action = z.repeat(p_logit.shape[0]*self.action_size,1)
        z_action = torch.reshape(z_action, (-1, self.action_size, self.num_support)).to(device)
        
        q_action = torch.sum(z_action * p_logit, dim=-1)
        
        return p_logit, q_action 