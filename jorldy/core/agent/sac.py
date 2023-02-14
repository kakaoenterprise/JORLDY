import torch

torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import os
import numpy as np

from core.network import Network
from core.optimizer import Optimizer
from core.buffer import ReplayBuffer
from .base import BaseAgent


class SAC(BaseAgent):
    """Soft actor critic (SAC) agent.
    Args:
        state_size (int): dimension of state.
        action_size (int): dimension of action.
        hidden_size (int): dimension of hidden unit.
        actor (str): key of actor network class in _network_dict.txt.
        critic (str): key of critic network class in _network_dict.txt.
        head (str): key of head in _head_dict.txt.
        optim_config (dict): dictionary of the optimizer info.
        use_dynamic_alpha (bool): parameter that determine whether to use autotunning entropy adjustment.
        gamma (float): discount factor.
        tau (float): the soft update coefficient (for soft target update).
        buffer_size (int): the size of the memory buffer.
        batch_size (int): the number of samples in the one batch.
        start_train_step (int): steps to start learning.
        static_log_alpha (float): static value used as log alpha when use_dynamic_alpha is false.
        target_update_period (int): period to update the target network (for hard target update) (unit: step)
        run_step (int): the number of total steps.
        lr_decay: lr_decay option which apply decayed weight on parameters of network.
        device (str): device to use.
            (e.g. 'cpu' or 'gpu'. None can also be used, and in this case, the cpu is used.)
    """

    def __init__(
        self,
        state_size,
        action_size,
        hidden_size=512,
        actor="continuous_policy",
        critic="continuous_q_network",
        head="mlp",
        optim_config={
            "actor": "adam",
            "critic": "adam",
            "alpha": "adam",
            "actor_lr": 5e-4,
            "critic_lr": 1e-3,
            "alpha_lr": 3e-4,
        },
        use_dynamic_alpha=False,
        gamma=0.99,
        tau=5e-3,
        buffer_size=50000,
        batch_size=64,
        start_train_step=2000,
        static_log_alpha=-2.0,
        target_update_period=10000,
        run_step=1e6,
        lr_decay=True,
        device=None,
        **kwargs,
    ):
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.action_type = actor.split("_")[0]

        self.actor = Network(
            actor, state_size, action_size, D_hidden=hidden_size, head=head
        ).to(self.device)
        self.actor_optimizer = Optimizer(
            optim_config["actor"], self.actor.parameters(), lr=optim_config["actor_lr"]
        )

        (
            self.critic1,
            self.target_critic1,
            self.critic_optimizer1,
        ) = self.critic_set(
            critic, state_size, action_size, hidden_size, head, optim_config
        )
        (
            self.critic2,
            self.target_critic2,
            self.critic_optimizer2,
        ) = self.critic_set(
            critic, state_size, action_size, hidden_size, head, optim_config
        )

        self.use_dynamic_alpha = use_dynamic_alpha
        if use_dynamic_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = Optimizer(
                optim_config["alpha"], [self.log_alpha], lr=optim_config["alpha_lr"]
            )
        else:
            self.log_alpha = torch.tensor(static_log_alpha).to(self.device)
            self.alpha_optimizer = None
        self.alpha = self.log_alpha.exp()

        if self.action_type == "continuous":
            self.target_entropy = -action_size
        else:
            self.target_entropy = -np.log(1 / action_size) * 0.98

        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.start_train_step = start_train_step
        self.run_step = run_step
        self.lr_decay = lr_decay
        self.num_learn = 0

        self.target_update_stamp = 0
        self.time_t = 0
        self.target_update_period = target_update_period

    def critic_set(
        self, critic_id, state_size, action_size, hidden_size, head, optim_config
    ):
        critic = Network(
            critic_id, state_size, action_size, D_hidden=hidden_size, head=head
        ).to(self.device)
        target_critic = Network(
            critic_id, state_size, action_size, D_hidden=hidden_size, head=head
        ).to(self.device)
        target_critic.load_state_dict(critic.state_dict())
        critic_optimizer = Optimizer(
            optim_config["critic"],
            critic.parameters(),
            lr=optim_config["critic_lr"],
        )
        return critic, target_critic, critic_optimizer

    @torch.no_grad()
    def act(self, state, training=True):
        self.actor.train(training)

        if self.action_type == "continuous":
            mu, std = self.actor(self.as_tensor(state))
            z = torch.normal(mu, std) if training else mu
            action = torch.tanh(z)
        else:
            pi = self.actor(self.as_tensor(state))
            action = (
                torch.multinomial(pi, 1)
                if training
                else torch.argmax(pi, dim=-1, keepdim=True)
            )
        action = action.cpu().numpy()
        return {"action": action}

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
        for key in transitions.keys():
            transitions[key] = self.as_tensor(transitions[key])

        state = transitions["state"]
        action = transitions["action"]
        reward = transitions["reward"]
        next_state = transitions["next_state"]
        done = transitions["done"]

        if self.action_type == "continuous":
            q1 = self.critic1(state, action)
            q2 = self.critic2(state, action)

            with torch.no_grad():
                mu, std = self.actor(next_state)
                next_action, next_log_prob = self.sample_action(mu, std)
                next_q1 = self.target_critic1(next_state, next_action)
                next_q2 = self.target_critic2(next_state, next_action)
                entropy = -next_log_prob
        else:
            q1 = self.critic1(state).gather(1, action.long())
            q2 = self.critic2(state).gather(1, action.long())

            with torch.no_grad():
                next_pi = self.actor(next_state)
                next_q1 = (next_pi * self.target_critic1(next_state)).sum(
                    -1, keepdim=True
                )
                next_q2 = (next_pi * self.target_critic2(next_state)).sum(
                    -1, keepdim=True
                )
                m = Categorical(next_pi)
                entropy = m.entropy().unsqueeze(-1)

        with torch.no_grad():
            min_next_q = torch.min(next_q1, next_q2)
            target_q = reward + (1 - done) * self.gamma * (
                min_next_q + self.alpha * entropy
            )

        max_Q = torch.max(target_q, axis=0).values.cpu().numpy()[0]

        # Critic
        critic_loss1 = F.mse_loss(q1, target_q)
        critic_loss2 = F.mse_loss(q2, target_q)

        self.critic_optimizer1.zero_grad(set_to_none=True)
        critic_loss1.backward()
        self.critic_optimizer1.step()

        self.critic_optimizer2.zero_grad(set_to_none=True)
        critic_loss2.backward()
        self.critic_optimizer2.step()

        # Actor
        if self.action_type == "continuous":
            mu, std = self.actor(state)
            sample_action, log_prob = self.sample_action(mu, std)
            q1 = self.critic1(state, sample_action)
            q2 = self.critic2(state, sample_action)
            entropy = -log_prob
        else:
            pi = self.actor(state)
            q1 = (pi * self.critic1(state)).sum(-1, keepdim=True)
            q2 = (pi * self.critic2(state)).sum(-1, keepdim=True)
            m = Categorical(pi)
            entropy = m.entropy().unsqueeze(-1)

        min_q = torch.min(q1, q2)
        actor_loss = -((self.alpha.detach() * entropy) + min_q).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha
        alpha_loss = self.log_alpha * (entropy - self.target_entropy).detach().mean()

        self.alpha = self.log_alpha.exp()

        if self.use_dynamic_alpha:
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.num_learn += 1

        result = {
            "critic_loss1": critic_loss1.item(),
            "critic_loss2": critic_loss2.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "max_Q": max_Q,
            "mean_Q": min_q.mean().item(),
            "alpha": self.alpha.item(),
            "entropy": entropy.mean().item(),
        }
        return result

    def update_target_soft(self):
        for t_p, p in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            t_p.data.copy_(self.tau * p.data + (1 - self.tau) * t_p.data)
        for t_p, p in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            t_p.data.copy_(self.tau * p.data + (1 - self.tau) * t_p.data)

    def update_target_hard(self):
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    def process(self, transitions, step):
        result = {}
        # Process per step
        self.memory.store(transitions)
        delta_t = step - self.time_t
        self.time_t = step
        self.target_update_stamp += delta_t

        if self.memory.size > self.batch_size and step >= self.start_train_step:
            result = self.learn()

            if self.lr_decay:
                self.learning_rate_decay(
                    step,
                    [
                        self.actor_optimizer,
                        self.critic_optimizer1,
                        self.critic_optimizer2,
                    ],
                )

        if self.num_learn > 0:
            if self.action_type == "continuous":
                self.update_target_soft()
            else:
                if self.target_update_stamp >= self.target_update_period:
                    self.update_target_hard()
                    self.target_update_stamp = 0

        return result

    def save(self, path):
        print(f"...Save model to {path}...")
        save_dict = {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic_optimizer1": self.critic_optimizer1.state_dict(),
            "critic_optimizer2": self.critic_optimizer2.state_dict(),
        }
        if self.use_dynamic_alpha:
            save_dict["log_alpha"] = self.log_alpha
            save_dict["alpha_optimizer"] = self.alpha_optimizer.state_dict()

        torch.save(save_dict, os.path.join(path, "ckpt"))

    def load(self, path):
        print(f"...Load model from {path}...")
        checkpoint = torch.load(os.path.join(path, "ckpt"), map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])

        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic1.load_state_dict(checkpoint["critic2"])
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer1.load_state_dict(checkpoint["critic_optimizer1"])
        self.critic_optimizer2.load_state_dict(checkpoint["critic_optimizer2"])

        if self.use_dynamic_alpha and "log_alpha" in checkpoint.keys():
            self.log_alpha = checkpoint["log_alpha"]
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])

    def sync_in(self, weights):
        self.actor.load_state_dict(weights)

    def sync_out(self, device="cpu"):
        weights = self.actor.state_dict()
        for k, v in weights.items():
            weights[k] = v.to(device)
        sync_item = {
            "weights": weights,
        }
        return sync_item
