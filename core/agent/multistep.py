import torch
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F

from core.buffer import MultistepBuffer
from .dqn import DQN

class Multistep(DQN):
    def __init__(self, n_step=5, **kwargs):
        super(Multistep, self).__init__(**kwargs)
        self.n_step = n_step
        self.memory = MultistepBuffer(self.buffer_size, self.n_step, self.num_worker)
    
    def learn(self):
#         shapes of 1-step implementations: (batch_size, dimension_data)
#         shapes of multistep implementations: (batch_size, steps, dimension_data)

        transitions = self.memory.sample(self.batch_size)
        for key in transitions.keys():
            transitions[key] = torch.as_tensor(transitions[key], dtype=torch.float32, device=self.device)

        state = transitions['state']
        action = transitions['action']
        reward = transitions['reward']
        next_state = transitions['next_state']
        done = transitions['done']
        
        eye = torch.eye(self.action_size).to(self.device)
        one_hot_action = eye[action[:, 0].view(-1).long()]
        q = (self.network(state) * one_hot_action).sum(1, keepdims=True)
        with torch.no_grad():
            max_Q = torch.max(q).item()
            next_q = self.target_network(next_state)
            target_q = next_q.max(1, keepdims=True).values

            for i in reversed(range(self.n_step)):
                target_q = reward[:, i] + (1 - done[:, i]) * self.gamma * target_q
            
        loss = F.smooth_l1_loss(q, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        
        self.num_learn += 1

        result = {
            "loss" : loss.item(),
            "epsilon" : self.epsilon,
            "max_Q": max_Q,
        }
        
        return result
    
    def process(self, transitions, step):
        result = {}

        # Process per step
        delta_t = step - self.time_t
        self.memory.store(transitions)
        self.time_t = step
        self.target_update_stamp += delta_t
        
        if self.memory.size >= self.batch_size and self.time_t >= self.start_train_step:
            result = self.learn()

        # Process per step if train start
        if self.num_learn > 0:
            self.epsilon_decay(delta_t)

            if self.target_update_stamp >= self.target_update_period:
                self.update_target()
                self.target_update_stamp = 0

        return result