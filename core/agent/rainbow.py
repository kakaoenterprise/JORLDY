import torch
import torch.nn.functional as F
import numpy as np
import copy

from core.network import Network
from core.optimizer import Optimizer
from .utils import RainbowBuffer
from .dqn import DQNAgent

class RainbowAgent(DQNAgent):
    def __init__(self,
                state_size,
                action_size,
                network='rainbow',
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
                # C51
                v_min = -10,
                v_max = 10,
                num_support = 51
                ):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size        
        self.network = Network(network, state_size, action_size, num_support, self.device).to(self.device)
        self.target_network = copy.deepcopy(self.network)
        self.optimizer = Optimizer(optimizer, self.network.parameters(), lr=learning_rate, eps=opt_eps)
        self.gamma = gamma
        self.explore_step = explore_step
        self.batch_size = batch_size
        self.start_train_step = start_train_step
        self.target_update_period = target_update_period
        # MultiStep
        self.n_step = n_step
        # PER
        self.alpha = alpha
        self.beta = beta
        self.learn_period = learn_period
        self.uniform_sample_prob = uniform_sample_prob
        self.beta_add = 1/self.explore_step
        # C51
        self.v_min = v_min
        self.v_max = v_max
        self.num_support = num_support
        
        # MultiStep
        self.memory = RainbowBuffer(buffer_size, self.n_step, self.uniform_sample_prob)
        
        # C51
        self.delta_z = (self.v_max - self.v_min) / (self.num_support - 1)
        self.z = torch.linspace(self.v_min, self.v_max, self.num_support, device=self.device).view(1, -1)
        
        self.num_learn = 0
        
    def act(self, state, training=True):
        self.network.train(training)
        
        if training and self.memory.size < max(self.batch_size, self.start_train_step):
            action = np.random.randint(0, self.action_size, size=(state.shape[0], 1))
        else:
            logits = self.network(torch.FloatTensor(state).to(self.device), training)
            _, q_action = self.logits2Q(logits)
            action = torch.argmax(q_action, -1, keepdim=True).data.cpu().numpy()
        return action

    def learn(self):
        if self.memory.buffer_counter < max(self.batch_size, self.start_train_step):
            return None

        transitions, weights, indices, sampled_p, mean_p = self.memory.sample(self.beta, self.batch_size)
        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(self.device), transitions)
        
        logit = self.network(state, True)
        p_logit, q_action = self.logits2Q(logit)
        
        action_eye = torch.eye(self.action_size).to(self.device)
        action_onehot = action_eye[action[:, 0].long()]
        
        p_action = torch.squeeze(action_onehot @ p_logit, 1)

        target_dist = torch.zeros(self.batch_size, self.num_support, device=self.device, requires_grad=False)
        with torch.no_grad():
            # Double
            _, next_q_action = self.logits2Q(self.network(next_state, False))
            
            target_p_logit, _ = self.logits2Q(self.target_network(next_state, False))
            
            target_action = torch.argmax(next_q_action, -1, keepdim=True)
            target_action_onehot = action_eye[target_action.long()]
            target_p_action = torch.squeeze(target_action_onehot @ target_p_logit, 1)
            
            for i in reversed(range(self.n_step)):
                Tz = reward[:, i].expand(-1,self.num_support) + (1 - done[:, i])*self.gamma*self.z
            
            b = torch.clamp(Tz - self.v_min, 0, self.v_max - self.v_min)/ self.delta_z
            l = torch.floor(b).long()
            u = torch.ceil(b).long()
            
            support_eye = torch.eye(self.num_support, device=self.device)
            l_support_onehot = support_eye[l]
            u_support_onehot = support_eye[u]

            l_support_binary = torch.unsqueeze(u-b, -1)
            u_support_binary = torch.unsqueeze(b-l, -1)
            target_p_action_binary = torch.unsqueeze(target_p_action, -1)
            
            lluu = l_support_onehot * l_support_binary + u_support_onehot * u_support_binary
                       
            target_dist += done[:,0,:] * torch.mean(l_support_onehot * u_support_onehot + lluu, 1)
            target_dist += (1 - done[:,0,:])* torch.sum(target_p_action_binary * lluu, 1)
            target_dist /= torch.clamp(torch.sum(target_dist, 1, keepdim=True), min=1e-8)

        max_Q = torch.max(q_action).item()
        max_logit = torch.max(logit).item()
        min_logit = torch.min(logit).item()
        
        # PER
        KL = -(target_dist*torch.clamp(p_action, min=1e-8).log()).sum(-1)
        
        p_j = torch.pow(KL, self.alpha)
        
        for i, p in zip(indices, p_j):
            self.memory.update_priority(p.item(), i)
        
        # Annealing beta
        self.beta = min(1.0, self.beta + self.beta_add)

        weights = torch.unsqueeze(torch.FloatTensor(weights).to(self.device), -1)
                    
        loss = (weights * KL).mean()
                
        self.optimizer.zero_grad()
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
    
    def process(self, state, action, reward, next_state, done):
        result = None
        # Process per step
        self.memory.store(state, action, reward, next_state, done)
        
        if self.memory.size > 0 and self.memory.size % self.learn_period == 0:
            result = self.learn()

        # Process per step if train start
        if self.num_learn > 0:
            if self.num_learn % self.target_update_period == 0:
                self.update_target()
        
        # Process per episode
        if done.all():
            pass
    
        return result
    
    def logits2Q(self, logits):
        logits_max = torch.max(logits, -1, keepdim=True).values
        p_logit = F.softmax(logits - logits_max, dim=-1)

        z_action = self.z.expand(p_logit.shape[0], self.action_size, self.num_support)
        q_action = torch.sum(z_action * p_logit, dim=-1)
        
        return p_logit, q_action