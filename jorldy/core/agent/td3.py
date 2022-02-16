import torch

torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
import os

from .ddpg import DDPG


class TD3(DDPG):
    action_type = "continuous"
    """Twin-delayed deep deterministic policy gradient (TD3) agent.

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
        mu (float): the drift coefficient of the Ornstein-Uhlenbeck process for action exploration.
        theta (float): reversion of the time constant of the Ornstein-Uhlenbeck process.
        sigma (float): diffusion coefficient of the Ornstein-Uhlenbeck process.
        device (str): device to use.
            (e.g. 'cpu' or 'gpu'. None can also be used, and in this case, the cpu is used.)
    """

    def __init__(
        self,
        actor="td3_actor",
        critic="td3_critic",
        actor_period=2,
        **kwargs,
    ):
        super(TD3, self).__init__(actor=actor, critic=critic, **kwargs)
        self.actor_period = actor_period
        self.actor_loss = 0.0

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
            next_actions = self.target_actor(next_state)
            next_q1, next_q2 = self.target_critic(next_state, next_actions)
            min_next_q = torch.min(next_q1, next_q2)
            target_q = reward + (1 - done) * self.gamma * min_next_q
        q = self.critic(state, action)
        critic_loss = F.mse_loss(target_q, q[0]) + F.mse_loss(target_q, q[1])
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        max_Q = torch.max(target_q, axis=0).values.cpu().numpy()[0]

        # Actor Update
        if not self.num_learn % self.actor_period:
            action_pred = self.actor(state)
            critic_1, _ = self.critic(state, action_pred)
            actor_loss = -critic_1.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor_loss = actor_loss.item()

            self.update_target_soft()

        self.num_learn += 1

        self.result = {
            "critic_loss": critic_loss.item(),
            "actor_loss": self.actor_loss,
            "max_Q": max_Q,
        }

        return self.result

    def process(self, transitions, step):
        result = {}
        # Process per step
        self.memory.store(transitions)

        if self.memory.size >= self.batch_size and step >= self.start_train_step:
            result = self.learn()
            self.learning_rate_decay(
                step, [self.actor_optimizer, self.critic_optimizer]
            )

        return result
