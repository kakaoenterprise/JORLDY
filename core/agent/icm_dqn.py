import torch
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
import numpy as np 

from .dqn import DQNAgent
from core.network import Network
from core.optimizer import Optimizer

class ICMDQNAgent(DQNAgent):
    def __init__(self,
                state_size,
                action_size,
                icm_network = 'icm',
                beta = 0.2,
                lamb = 1.0,
                eta = 0.01,
                extrinsic_coeff = 1.0,
                intrinsic_coeff = 0.01,
                **kwargs,
                ):
        super(ICMDQNAgent, self).__init__(state_size, action_size, **kwargs)
        
        self.icm = Network(icm_network, state_size, action_size, eta, 'discrete').to(self.device)
        
        parameters = list(self.network.parameters()) + list(self.icm.parameters())
        self.optimizer = Optimizer('adam', parameters, lr=self.learning_rate, eps=self.opt_eps)
        
        self.beta = beta
        self.lamb = lamb
        self.eta = eta
        self.extrinsic_coeff = extrinsic_coeff
        self.intrinsic_coeff = intrinsic_coeff
    
    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training)
        
        if training and self.memory.size < max(self.batch_size, self.start_train_step):
            action = np.random.randint(0, self.action_size, size=(state.shape[0], 1))
        else:
            action = torch.argmax(self.network(torch.FloatTensor(state).to(self.device)), -1, keepdim=True).data.cpu().numpy()
        
        return action
    
                                  
    def learn(self):        
        transitions = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(self.device), transitions)
        
        eye = torch.eye(self.action_size).to(self.device)
        one_hot_action = eye[action.view(-1).long()]
        q = (self.network(state) * one_hot_action).sum(1, keepdims=True)
        
        #ICM 
        r_i, l_f, l_i = self.icm(state, action, next_state)
        reward = (self.extrinsic_coeff * reward) + (self.intrinsic_coeff * r_i.unsqueeze(1))
        
        with torch.no_grad():
            max_Q = torch.max(q).item()
            next_q = self.target_network(next_state)
            target_q = reward + (1 - done) * self.gamma * next_q.max(1, keepdims=True).values
                
        # ICM
        loss_origin = F.smooth_l1_loss(q, target_q)
        loss = self.lamb * loss_origin + (self.beta * l_f) + ((1-self.beta) * l_i)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.num_learn += 1
        
        result = {
            "loss" : loss_origin.item(),
            "max_Q": max_Q,
            "r_i": r_i.mean().item(),
            'l_f': l_f.item(),
            'l_i': l_i.item(),
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
    
    def save(self, path):
        print(f"...Save model to {path}...")
        torch.save({
            "network" : self.network.state_dict(),
            "icm" : self.icm.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
        }, os.path.join(path,"ckpt"))
        
    def get_ri(self, state, next_state, action):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)

        r_i, _, _ = self.icm(state, action, next_state)
        
        return r_i