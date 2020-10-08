import torch
import torch.nn.functional as F
from torch.distributions import Normal

from core.utils import ReplayBuffer

class SACAgent:
    def __init__(self,
                actor,
                critic,
                target_critic,
                actor_optimizer,
                critic_optimizer,
                use_dynamic_alpha = False,
                log_alpha = None,
                alpha_optimizer = None,
                gamma=0.99,
                tau=5e-3,
                buffer_size=50000,
                batch_size = 64,
                start_train=2000,
                static_log_alpha=-2.0
                ):
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.use_dynamic_alpha = use_dynamic_alpha
        self.log_alpha = log_alpha if use_dynamic_alpha else torch.tensor(static_log_alpha)
        self.alpha = self.log_alpha.exp()
        
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.alpha_optimizer = alpha_optimizer if use_dynamic_alpha else None
        self.target_entropy = -torch.prod(torch.Tensor(actor.D_out)).item()

        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.start_train = start_train

        self.update_target('hard')

    def act(self, state, training=True):
        mu, std = self.actor(state)
        std = std if training else 0
        m = Normal(mu, std)
        z = m.rsample()
        action = torch.tanh(z).item()
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
        if self.memory.size < max(self.batch_size, self.start_train):
            return 0

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        q1, q2 = self.critic(state, action)

        with torch.no_grad():
            mu, std = self.actor(next_state)
            next_action, next_log_prob = self.sample_action(mu, std)
            next_target_q1, next_target_q2 = self.target_critic(next_state, next_action)
            min_next_target_q = torch.min(next_target_q1, next_target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done)*self.gamma*min_next_target_q
        
        max_Q = torch.mean(torch.max(target_q, axis=0).values).values
        
        # Critic
        critic_loss1 = F.mse_loss(q1, target_q)
        critic_loss2 = F.mse_loss(q2, target_q)
        critic_loss = critic_loss1 + critic_loss2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor
        mu, std = self.actor(state)
        action, log_prob = self.sample_action(mu, std)

        q1, q2 = self.critic(state, action)
        min_q = torch.min(q1, q2)

        actor_loss = ((self.alpha * log_prob) - min_q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        if self.use_dynamic_alpha:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

        self.update_target('soft')
        
        return critic_loss1.item(), critic_loss2.item(), actor_loss.item(), alpha_loss.item(), max_Q, self.alpha.item()

    def update_target(self, mode):
        if mode=='hard':  
            self.target_critic.load_state_dict(self.critic.state_dict())
        elif mode=='soft':
            for t_p, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                t_p.data.copy_(self.tau*p.data + (1-self.tau)*t_p.data)

    def remember(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)