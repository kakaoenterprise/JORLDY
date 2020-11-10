import torch
import torch.nn.functional as F
import random
import os

from .dqn import DQNAgent
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DoubleDQNAgent(DQNAgent):
    def __init__(self, **kwargs):
        super(DoubleDQNAgent, self).__init__(**kwargs)

    def learn(self):        
        if self.memory.length < max(self.batch_size, self.start_train_step):
            return None
        
        transitions = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device), transitions)
        
        eye = torch.eye(self.action_size).to(device)
        one_hot_action = eye[action.view(-1).long()]
        q = (self.network(state) * one_hot_action).sum(1, keepdims=True)
        
        with torch.no_grad():
            max_Q = torch.max(q).item()
            next_q = self.network(next_state)
            max_a = torch.argmax(next_q, axis=1)
            max_eye = torch.eye(self.action_size).to(device)
            max_one_hot_action = eye[max_a.view(-1).long()]
            
            next_target_q = self.target_network(next_state)
            target_q = reward + (next_target_q * max_one_hot_action).sum(1, keepdims=True) * (self.gamma*(1 - done))
        
        loss = F.smooth_l1_loss(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.num_learn % self.target_update_term == 0:
            self.update_target()
        self.num_learn += 1
        
        result = {
            "loss" : loss.item(),
            "epsilon" : self.epsilon,
            "max_Q": max_Q,
        }
        return result

        predict_Q = self.model(torch.FloatTensor(state_batch).to(self.device))
        target_Q = predict_Q.cpu().detach().numpy()
        target_nextQ = self.target_model(torch.FloatTensor(next_state_batch).to(self.device)).cpu().detach().numpy()

        Q_a = self.model(torch.FloatTensor(next_state_batch).to(self.device)).cpu().detach().numpy()
        max_Q = np.max(target_Q)

        with torch.no_grad():
            for i in range(config.batch_size):
                if done_batch[i]:
                    target_Q[i, action_batch[i]] = reward_batch[i]
                else:
                    action_ind = np.argmax(Q_a[i])
                    target_Q[i, action_batch[i]] = reward_batch[i] + config.discount_factor * target_nextQ[i][action_ind]