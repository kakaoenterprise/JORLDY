import torch

torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
import os

from core.network import Network
from core.optimizer import Optimizer
from core.buffer import ReplayBuffer
from .base import BaseAgent
from .utils import OU_Noise


class DDPG(BaseAgent):
    action_type = "continuous"
    """Deep deterministic policy gradient (DDPG) agent.

    Args:
        state_size (int): dimension of state.
        action_size (int): dimension of action.
        hidden_size (int): dimension of hidden unit.
        actor (str): key of actor network class in _network_dict.txt.
        critic (str): key of critic network class in _network_dict.txt.
        head (str): key of head in _head_dict.txt.
        optim_config (dict): dictionary of the optimizer info.
        gamma (float): discount factor.
        buffer_size (int): the size of the memory buffer.
        batch_size (int): the number of samples in the one batch.
        start_train_step (int): steps to start learning.
        tau (float): the soft update coefficient.
        run_step (int): the number of total steps.
        lr_decay: lr_decay option which apply decayed weight on parameters of network.
        mu (float): the drift coefficient of the Ornstein-Uhlenbeck process for action exploration.
        theta (float): reversion of the time constant of the Ornstein-Uhlenbeck process.
        sigma (float): diffusion coefficient of the Ornstein-Uhlenbeck process.
        device (str): device to use.
            (e.g. 'cpu' or 'gpu'. None can also be used, and in this case, the cpu is used.)
    """

    def __init__(
        self,
        state_size,
        action_size,
        hidden_size=512,
        actor="deterministic_policy",
        critic="continuous_q_network",
        head="mlp",
        optim_config={
            "actor": "adam",
            "critic": "adam",
            "actor_lr": 5e-4,
            "critic_lr": 1e-3,
        },
        gamma=0.99,
        buffer_size=50000,
        batch_size=128,
        start_train_step=2000,
        tau=1e-3,
        run_step=1e6,
        lr_decay=True,
        # OU noise
        mu=0,
        theta=1e-3,
        sigma=2e-3,
        device=None,
        **kwargs,
    ):
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.actor = Network(
            actor, state_size, action_size, D_hidden=hidden_size, head=head
        ).to(self.device)
        self.critic = Network(
            critic, state_size, action_size, D_hidden=hidden_size, head=head
        ).to(self.device)
        self.target_actor = Network(
            actor, state_size, action_size, D_hidden=hidden_size, head=head
        ).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic = Network(
            critic, state_size, action_size, D_hidden=hidden_size, head=head
        ).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = Optimizer(
            optim_config["actor"], self.actor.parameters(), lr=optim_config["actor_lr"]
        )
        self.critic_optimizer = Optimizer(
            optim_config["critic"],
            self.critic.parameters(),
            lr=optim_config["critic_lr"],
        )

        self.OU = OU_Noise(action_size, mu, theta, sigma)

        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.start_train_step = start_train_step
        self.num_learn = 0
        self.run_step = run_step
        self.lr_decay = lr_decay

    @torch.no_grad()
    def act(self, state, training=True):
        self.actor.train(training)
        mu = self.actor(self.as_tensor(state))
        mu = mu.cpu().numpy()
        action = mu + self.OU.sample().clip(-1.0, 1.0) if training else mu
        return {"action": action}

    def learn(self):
        transitions = self.memory.sample(self.batch_size)
        for key in transitions.keys():
            transitions[key] = self.as_tensor(transitions[key])

        state = transitions["state"]
        action = transitions["action"]
        reward = transitions["reward"]
        next_state = transitions["next_state"]
        done = transitions["done"]

        # Critic Update
        with torch.no_grad():
            next_action = self.target_actor(next_state)
            next_q = self.target_critic(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * next_q
        q = self.critic(state, action)
        critic_loss = F.mse_loss(target_q, q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        max_Q = torch.max(target_q, axis=0).values.cpu().numpy()[0]

        # Actor Update
        action_pred = self.actor(state)
        actor_loss = -self.critic(state, action_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.num_learn += 1

        result = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "max_Q": max_Q,
        }
        return result

    def update_target_soft(self):
        for t_p, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            t_p.data.copy_(self.tau * p.data + (1 - self.tau) * t_p.data)
        for t_p, p in zip(self.target_actor.parameters(), self.actor.parameters()):
            t_p.data.copy_(self.tau * p.data + (1 - self.tau) * t_p.data)

    def process(self, transitions, step):
        result = {}
        # Process per step
        self.memory.store(transitions)

        if self.memory.size >= self.batch_size and step >= self.start_train_step:
            result = self.learn()
            if self.lr_decay:
                self.learning_rate_decay(
                    step, [self.actor_optimizer, self.critic_optimizer]
                )
        if self.num_learn > 0:
            self.update_target_soft()

        return result

    def save(self, path):
        print(f"...Save model to {path}...")
        save_dict = {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
        torch.save(save_dict, os.path.join(path, "ckpt"))

    def load(self, path):
        print(f"...Load model from {path}...")
        checkpoint = torch.load(os.path.join(path, "ckpt"), map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])

        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

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
