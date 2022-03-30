import torch

torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
import numpy as np
import os

from core.network import Network
from core.optimizer import Optimizer
from core.buffer import ReplayBuffer
from .base import BaseAgent


class DQN(BaseAgent):
    action_type = "discrete"
    """DQN agent.

    Args:
        state_size (int): dimension of state.
        action_size (int): dimension of action.
        hidden_size (int): dimension of hidden unit.
        optim_config (dict): dictionary of the optimizer info.
            (key: 'name', value: name of optimizer)
        network (str): key of network class in _network_dict.txt.
        head (str): key of head in _head_dict.txt.
        gamma (float): discount factor.
        epsilon_init (float): initial epsilon value (random action ratio) in decaying epsilon-greedy policy.
        epsilon_min (float): final epsilon value in decaying epsilon-greedy policy.
        epsilon_eval (float): evaluate time epsilon value.
        explore_ratio (float): the ratio of steps the epsilon decays.
        buffer_size (int): the size of the memory buffer.
        batch_size (int): the number of samples in the one batch.
        start_train_step (int): steps to start learning.
        target_update_period (int): period to update the target network (unit: step)
        device (str): device to use.
            (e.g. 'cpu' or 'gpu'. None can also be used, and in this case, the cpu is used.)
        run_step (int): the number of total steps.
        num_workers: the number of agents in distributed learning.
        lr_decay: lr_decay option which apply decayed weight on parameters of network.
    """

    def __init__(
        self,
        state_size,
        action_size,
        hidden_size=512,
        optim_config={"name": "adam"},
        network="discrete_q_network",
        head="mlp",
        gamma=0.99,
        epsilon_init=1.0,
        epsilon_min=0.1,
        epsilon_eval=0.0,
        explore_ratio=0.1,
        buffer_size=50000,
        batch_size=64,
        start_train_step=2000,
        target_update_period=500,
        device=None,
        run_step=1e6,
        num_workers=1,
        lr_decay=True,
        **kwargs,
    ):
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.action_size = action_size
        self.action_type = "discrete"
        self.network = Network(
            network, state_size, action_size, D_hidden=hidden_size, head=head
        ).to(self.device)
        self.target_network = Network(
            network, state_size, action_size, D_hidden=hidden_size, head=head
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = Optimizer(**optim_config, params=self.network.parameters())
        self.gamma = gamma
        self.epsilon = epsilon_init
        self.epsilon_init = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_eval = epsilon_eval
        self.explore_step = run_step * explore_ratio
        self.epsilon_delta = (epsilon_init - epsilon_min) / self.explore_step
        self.buffer_size = buffer_size
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.start_train_step = start_train_step
        self.target_update_stamp = 0
        self.target_update_period = target_update_period
        self.num_learn = 0
        self.time_t = 0
        self.num_workers = num_workers
        self.run_step = run_step
        self.lr_decay = lr_decay

    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training)
        epsilon = self.epsilon if training else self.epsilon_eval

        if np.random.random() < epsilon:
            batch_size = (
                state[0].shape[0] if isinstance(state, list) else state.shape[0]
            )
            action = np.random.randint(0, self.action_size, size=(batch_size, 1))
        else:
            action = (
                torch.argmax(self.network(self.as_tensor(state)), -1, keepdim=True)
                .cpu()
                .numpy()
            )
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

        eye = torch.eye(self.action_size, device=self.device)
        one_hot_action = eye[action.view(-1).long()]
        q = (self.network(state) * one_hot_action).sum(1, keepdims=True)
        with torch.no_grad():
            max_Q = torch.max(q).item()
            next_q = self.target_network(next_state)
            target_q = (
                reward + (1 - done) * self.gamma * next_q.max(1, keepdims=True).values
            )

        loss = F.smooth_l1_loss(q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.num_learn += 1

        result = {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "max_Q": max_Q,
        }

        return result

    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def process(self, transitions, step):
        result = {}

        # Process per step
        self.memory.store(transitions)
        delta_t = step - self.time_t
        self.time_t = step
        self.target_update_stamp += delta_t

        if self.memory.size >= self.batch_size and self.time_t >= self.start_train_step:
            result = self.learn()
            if self.lr_decay:
                self.learning_rate_decay(step)

        # Process per step if train start
        if self.num_learn > 0:
            self.epsilon_decay(delta_t)

            if self.target_update_stamp >= self.target_update_period:
                self.update_target()
                self.target_update_stamp -= self.target_update_period

        return result

    def epsilon_decay(self, delta_t):
        new_epsilon = self.epsilon - delta_t * self.epsilon_delta
        self.epsilon = max(self.epsilon_min, new_epsilon)

    def save(self, path):
        print(f"...Save model to {path}...")
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            os.path.join(path, "ckpt"),
        )

    def load(self, path):
        print(f"...Load model from {path}...")
        checkpoint = torch.load(os.path.join(path, "ckpt"), map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.target_network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def set_distributed(self, id):
        self.epsilon = id / self.num_workers
        return self
