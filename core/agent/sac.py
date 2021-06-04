import torch
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
from torch.distributions import Normal
import os 
import copy

from core.network import Network
from core.optimizer import Optimizer
from .utils import ReplayBuffer
from .base import BaseAgent

class SACAgent(BaseAgent):
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
                device=None,
                ):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Network(actor, state_size, action_size).to(self.device)
        self.critic = Network(critic, state_size+action_size, action_size).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
        self.actor_optimizer = Optimizer(actor_optimizer, self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Optimizer(critic_optimizer, self.critic.parameters(), lr=critic_lr)
        
        self.use_dynamic_alpha = use_dynamic_alpha
        if use_dynamic_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = Optimizer(alpha_optimizer, [self.log_alpha], lr=alpha_lr)
        else:
            self.log_alpha = torch.tensor(static_log_alpha).to(self.device)
            self.alpha_optimizer = None
        self.alpha = self.log_alpha.exp()
        self.target_entropy = -1.0

        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.start_train_step = start_train_step
        self.num_learn = 0

    @torch.no_grad()
    def act(self, state, training=True):
        self.actor.train(training)
        mu, std = self.actor(torch.as_tensor(state, dtype=torch.float32, device=self.device))
        std = std if training else torch.zeros_like(std, device=self.device) + 1e-4
        m = Normal(mu, std)
        z = m.sample()
        action = torch.tanh(z)
        action = action.cpu().numpy()
        return action

    def sample_action(self, mu, std):
        m = Normal(mu, std)
        z = m.rsample()
        action = torch.tanh(z)
        log_prob = m.log_prob(z)
        # Enforcing Action Bounds
        log_prob -= torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

    def learn(self):
        transitions = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = map(lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device), transitions)

        q1, q2 = self.critic(state, action)

        with torch.no_grad():
            mu, std = self.actor(next_state)
            next_action, next_log_prob = self.sample_action(mu, std)
            next_q1, next_q2 = self.target_critic(next_state, next_action)
            min_next_q = torch.min(next_q1, next_q2)
            target_q = reward + (1 - done)*self.gamma*(min_next_q - self.alpha*next_log_prob)
        
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

        actor_loss = ((self.alpha.to(self.device) * log_prob) - min_q).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha = self.log_alpha.exp()
            
        if self.use_dynamic_alpha:
            self.alpha_optimizer.zero_grad(set_to_none=True)
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

    def update_target_soft(self):
        for t_p, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            t_p.data.copy_(self.tau*p.data + (1-self.tau)*t_p.data)
    
    def process(self, transitions, step):
        result = {}
        # Process per step
        self.memory.store(transitions)
        
        if self.memory.size > self.batch_size and step >= self.start_train_step:
            result = self.learn()
        if self.num_learn > 0:
            self.update_target_soft()

        return result

    def save(self, path):
        print(f"...Save model to {path}...")
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
        print(f"...Load model from {path}...")
        checkpoint = torch.load(os.path.join(path,"ckpt"),map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])

        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        
        if self.use_dynamic_alpha and 'log_alpha' in checkpoint.keys():
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            
    def sync_in(self, weights):
        self.actor.load_state_dict(weights)
    
    def sync_out(self, device="cpu"):
        weights = self.actor.state_dict()
        for k, v in weights.items():
            weights[k] = v.to(device) 
        sync_item ={
            "weights": weights,
        }
        return sync_item