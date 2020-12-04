import torch
import torch.nn.functional as F
from torch.distributions import Normal

from core.network import Network
from core.optimizer import Optimizer
from .utils import ReplayBuffer

import os 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SACAgent:
    def __init__(self,
                 state_size,
                 action_size,
                 actor = "sac_actor",
                 critic = "sac_critic",
                 actor_optimizer = "adam",
                 critic_optimizer = "adam",
                 alpha_optimizer = "adam",
                 actor_lr = 5e-4,
                 critic_lr = 1e-3,
                 alpha_lr = 3e-4,
                 use_dynamic_alpha = False,
                 gamma=0.99,
                 tau=5e-3,
                 buffer_size=50000,
                 batch_size = 64,
                 start_train_step=2000,
                 static_log_alpha=-2.0,
                 ):
        self.actor = Network(actor, state_size, action_size).to(device)
        self.critic = Network(critic, state_size+action_size, action_size).to(device)
        self.target_critic = Network(critic, state_size+action_size, action_size).to(device)
        self.actor_optimizer = Optimizer(actor_optimizer, self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Optimizer(critic_optimizer, self.critic.parameters(), lr=critic_lr)
        
        self.use_dynamic_alpha = use_dynamic_alpha
        if use_dynamic_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = Optimizer(alpha_optimizer, [self.log_alpha], lr=alpha_lr)
        else:
            self.log_alpha = torch.tensor(static_log_alpha).to(device)
            self.alpha_optimizer = None
        self.alpha = self.log_alpha.exp()
        self.target_entropy = -torch.prod(torch.Tensor(action_size)).to(device).item()

        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.start_train_step = start_train_step
        self.num_learn = 0

        self.update_target('hard')

    def act(self, state, training=True):
        mu, std = self.actor(torch.FloatTensor(state).to(device))
        std = std if training else 0
        m = Normal(mu, std)
        z = m.rsample()
        action = torch.tanh(z)
        action = action[0].data.cpu().detach().numpy()
        return action

    def sample_action(self, mu, std):
        m = Normal(mu, std)
        z = m.rsample()
        action = torch.tanh(z)
        log_prob = m.log_prob(z)
        # Enforcing Action Bounds
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

    def learn(self):
        if self.memory.length < max(self.batch_size, self.start_train_step):
            return None

        transitions = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device), transitions)

        q1, q2 = self.critic(state, action)

        with torch.no_grad():
            mu, std = self.actor(next_state)
            next_action, next_log_prob = self.sample_action(mu, std)
            next_target_q1, next_target_q2 = self.target_critic(next_state, next_action)
            min_next_target_q = torch.min(next_target_q1, next_target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done)*self.gamma*min_next_target_q
        
        max_Q = torch.max(target_q, axis=0).values.cpu().numpy()[0]
        
        # Critic
        critic_loss1 = F.mse_loss(q1, target_q)
        critic_loss2 = F.mse_loss(q2, target_q)
        critic_loss = critic_loss1 + critic_loss2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor
        mu, std = self.actor(state)
        sample_action, log_prob = self.sample_action(mu, std)

        q1, q2 = self.critic(state, sample_action)
        min_q = torch.min(q1, q2)

        actor_loss = ((self.alpha.to(device) * log_prob) - min_q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha = self.log_alpha.exp()
            
        if self.use_dynamic_alpha:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.num_learn += 1
        
        result = {
            'critic_loss1' : critic_loss1.item(),
            'critic_loss2' : critic_loss2.item(),
            'actor_loss' : actor_loss.item(),
            'alpha_loss' : alpha_loss.item(),
            'max_Q' : max_Q,
            'alpha' : self.alpha.item(),
        }
        return result

    def update_target(self, mode):
        if mode=='hard':  
            self.target_critic.load_state_dict(self.critic.state_dict())
        elif mode=='soft':
            for t_p, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                t_p.data.copy_(self.tau*p.data + (1-self.tau)*t_p.data)
    
    def process(self, state, action, reward, next_state, done):
        result = None
        # Process per step
        self.memory.store(state, action, reward, next_state, done)        
        result = self.learn()
        if self.num_learn > 0:
            self.update_target('soft')

        # Process per epi
        if done :
            pass

        return result

    def save(self, path):
        save_dict = {
            "actor" : self.actor.state_dict(),
            "actor_optimizer" : self.actor_optimizer.state_dict(),
            "critic" : self.critic.state_dict(),
            "critic_optimizer" : self.critic_optimizer.state_dict(),
        }
        if self.use_dynamic_alpha:
            save_dict['log_alpha'] = self.log_alpha
            save_dict['alpha_optimizer'] = self.alpha_optimizer.state_dict()

        torch.save(save_dict, os.path.join(path,"ckpt"))

    def load(self, path):
        checkpoint = torch.load(os.path.join(path,"ckpt"),map_location=device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])

        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        
        self.update_target('hard')

        if self.use_dynamic_alpha and 'log_alpha' in checkpoint.keys():
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])


