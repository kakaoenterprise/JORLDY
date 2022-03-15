from collections import defaultdict, deque
from collections.abc import Iterable
import os
import torch
import torch.nn.functional as F
import numpy as np

torch.backends.cudnn.benchmark = True

from core.network import Network
from core.optimizer import Optimizer
from core.buffer import ReplayBuffer
from core.buffer import PERBuffer
from .base import BaseAgent


class MuZero(BaseAgent):
    action_type = "discrete"
    """MuZero agent.

    Args:
        -
    """

    def __init__(
        self,
        # MuZero
        device=None,
        network="pseudo",
        state_size=(1, 1, 96, 96),
        hidden_state_channel=4,
        action_size=18,
        gamma=0.99,
        batch_size=16,
        start_train_step=0,
        trajectory_size=200,
        value_loss_weight=0.25,
        num_simulation=50,
        num_unroll=5,
        num_td_step=10,
        num_stack=32,
        buffer_size=10000,
        run_step=1e6,
        optim_config={
            "name": "adam",
            "lr": 5e-4,
        },
        # PER
        alpha=0.6,
        beta=0.4,
        learn_period=1,
        uniform_sample_prob=1e-3,
        **kwargs,
    ):
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        if not isinstance(state_size, Iterable):
            state_size = (1, state_size, 8, 8)
        stacked_shape = (
            (state_size[0] + 1) * num_stack + state_size[0],
            *state_size[1:],
        )
        self.network = PseudoNetwork(stacked_shape, hidden_state_channel, action_size)
        self.optimizer = Optimizer(
            optim_config["name"], self.network.parameters(), lr=optim_config["lr"]
        )

        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.start_train_step = start_train_step
        self.value_loss_weight = value_loss_weight

        self.trajectory_size = trajectory_size
        self.num_simulation = num_simulation
        self.num_unroll = num_unroll
        self.num_td_step = num_td_step
        self.num_stack = num_stack

        self.time_t = 0
        self.trajectory_step_stamp = 0
        self.run_step = run_step
        self.num_learn = 0

        self.trajectory = None

        # PER
        # self.alpha = alpha
        # self.beta = beta
        self.learn_period = learn_period
        self.learn_period_stamp = 0
        # self.uniform_sample_prob = uniform_sample_prob
        # self.beta_add = (1 - beta) / run_step
        # self.buffer_size = buffer_size
        # self.memory = PERBuffer(self.buffer_size, uniform_sample_prob)
        # no priority
        self.memory = ReplayBuffer(buffer_size)
        self.memory.first_store = False

        # MCTS
        self.mcts = MCTS(
            self.network.prediction,
            self.network.dynamics,
            self.action_size,
            self.num_simulation,
            self.num_unroll,
            self.gamma,
        )

    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training)

        if not self.trajectory:
            self.trajectory = Trajectory(state)

        states, actions = self.trajectory.get_stacked_data(
            self.trajectory_step_stamp, self.num_stack
        )
        states = self.as_tensor(np.expand_dims(states, axis=0))
        actions = self.as_tensor(np.expand_dims(actions, axis=0))
        root_state = self.network.representation(states, actions)
        if self.network == "pseudo":
            action, pi, value = self.mcts.run_mcts(root_state)
        else:
            pi = np.ones(self.action_size) / self.action_size
            action = np.random.choice(self.action_size, size=(1, 1))
        value = np.random.random()

        return {"action": action, "value": value, "pi": pi}

    def learn(self):
        trajectories = self.memory.sample(self.batch_size)

        transitions = defaultdict(list)
        for _, trajectory in enumerate(trajectories["trajectory"]):
            trajectory_len = len(trajectory.values)
            start_idx = np.random.choice(trajectory_len)
            last_idx = start_idx + self.num_unroll + 1
            values = []
            policies = trajectory.policies[start_idx:last_idx]
            rewards = trajectory.rewards[start_idx:last_idx]
            actions = trajectory.actions[start_idx:last_idx]
            for idx in range(start_idx, last_idx):
                if idx < trajectory_len:
                    values.append(
                        trajectory.get_bootstrap_value(
                            idx, self.num_td_step, self.gamma
                        )
                    )
                else:
                    values.append(np.zeros((1, 1)))
                    policies.append(np.ones(self.action_size) / self.action_size)
                    if idx > trajectory_len:
                        rewards.append(np.zeros((1, 1)))
                        actions.append(np.random.choice(self.action_size, size=(1, 1)))

            transitions["value"].append(values)
            transitions["policy"].append(policies)
            transitions["reward"].append(rewards)
            transitions["action"].append(actions)
            states, actions = trajectory.get_stacked_data(start_idx, self.num_stack)
            transitions["states"].append(states)
            transitions["actions"].append(actions)

        for key in transitions.keys():
            a = np.stack(transitions[key], axis=0)
            if a.shape[-1] == 1:
                a = a.squeeze(axis=-1)
            transitions[key] = self.as_tensor(a)

        states = transitions["states"]
        actions = transitions["actions"]
        target_action = transitions["action"]
        target_reward = transitions["reward"]
        target_policy = transitions["policy"]
        target_value = transitions["value"]

        hidden_state = self.network.representation(states, actions)
        pi, value = self.network.prediction(hidden_state)
        reward = torch.zeros(target_reward.shape)

        policy_loss = (-target_policy[:, 0] * torch.nn.LogSoftmax(dim=1)(pi)).sum(1)
        # value_loss = (-target_value[:, 0] * torch.nn.LogSoftmax(dim=1)(value)).sum(1)
        value_loss = F.smooth_l1_loss(target_value[:, 0], value)
        F.smooth_l1_loss(target_reward[:, 0], torch.zeros_like(target_reward[:, 0]))
        # (-target_reward[:, 0] * torch.nn.LogSoftmax(dim=1)(reward)).sum(1)
        reward_loss = 0

        for i in range(1, self.num_unroll):
            hidden_state, reward = self.network.dynamics(
                hidden_state, target_action[:, i]
            )
            pi, value = self.network.prediction(hidden_state)
            policy_loss += (-target_policy[:, i] * torch.nn.LogSoftmax(1)(pi)).sum(1)
            # value_loss += (-target_value[:, i] * torch.nn.LogSoftmax(1)(value)).sum(1)
            # reward_loss += (-target_reward[:, i] * torch.nn.LogSoftmax(1)(reward)).sum(1)
            value_loss = F.smooth_l1_loss(target_value[:, i], value)
            reward_loss = F.smooth_l1_loss(target_reward[:, i], reward)

        loss = (self.value_loss_weight * value_loss + policy_loss + reward_loss).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.num_learn += 1

        result = {
            "loss": loss.item(),
            "policy_loss": policy_loss.mean().item(),
            "value_loss": value_loss.item(),
            "reward_loss": reward_loss.item(),
        }

        return result

    def process(self, transitions, step):
        result = {}

        if self.memory.buffer_counter or transitions:
            # Process per step
            delta_t = step - self.time_t
            self.memory.store(transitions)
            self.time_t = step
            self.learn_period_stamp += delta_t

            if (
                self.learn_period_stamp >= self.learn_period
                and self.memory.buffer_counter >= self.batch_size
                and self.time_t >= self.start_train_step
            ):
                result = self.learn()
                self.learning_rate_decay(step)
                self.learn_period_stamp = 0

        return result

    def interact_callback(self, transition):
        # TODO: what if the trajectory is already terminal?
        _transition = None
        self.trajectory_step_stamp += 1

        self.trajectory.append(transition)

        if self.trajectory_step_stamp >= self.trajectory_size:
            self.trajectory_step_stamp = 0  # never self.trajectory_step_stamp -= period
            _transition = {"trajectory": [self.trajectory]}
            self.trajectory = None

        return _transition

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
        self.optimizer.load_state_dict(checkpoint["optimizer"])


class PseudoNetwork(torch.nn.Module):
    def __init__(self, s_shape, hidden_out, action_size):
        super().__init__()
        self.encode = torch.nn.Conv2d(s_shape[0], hidden_out, s_shape[1] - 1)
        self.hidden_flatten = torch.nn.Flatten()
        flatten_size = hidden_out * ((1 // 1 + 1) ** 2)
        self.pi = torch.nn.Linear(in_features=flatten_size, out_features=action_size)
        self.value = torch.nn.Linear(in_features=flatten_size, out_features=1)
        self.reward = torch.nn.Linear(in_features=flatten_size, out_features=1)
        self.next_hidden = torch.nn.Conv2d(hidden_out, hidden_out, 1)

    def representation(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        return self.encode(x)

    def prediction(self, hidden_state):
        x = self.hidden_flatten(hidden_state)
        return self.pi(x), self.value(x)

    def dynamics(self, hidden_state, action):
        x = self.hidden_flatten(hidden_state)
        return self.next_hidden(hidden_state), self.reward(x)


class MCTS:
    def __init__(self, p_fn, d_fn, action_size, n_mcts, n_unroll, gamma):
        self.p_fn = p_fn  # prediction function
        self.d_fn = d_fn  # dynamics function

        self.action_size = action_size
        self.n_mcts = n_mcts
        self.n_unroll = n_unroll + 1

        self.gamma = gamma
        self.temp_param = 1.0

        self.c1 = 1.25
        self.c2 = 19652
        self.alpha = 0.3

        self.q_min = 0
        self.q_max = 0

        self.root_id = (0,)
        self.tree = {}

    def run_mcts(self, root_state):
        self.tree = self.init_mcts(root_state)

        for i in range(self.n_mcts):
            # selection
            leaf_id, leaf_state = self.selection(root_state)

            # expansion and evaluation
            leaf_v = self.expansion(leaf_id, leaf_state)

            # backup
            self.backup(leaf_id, leaf_v)

        root_value = self.tree[self.root_id]["q"]
        root_action, pi = self.select_root_action()

        return root_action, pi, root_value

    def selection(self, root_state):
        node_id = self.root_id
        node_state = root_state

        while self.tree[node_id]["n"] > 0:
            if len(node_id) <= self.n_unroll:
                UCB_list = []
                total_n = self.tree[node_id]["n"]

                # for action_idx in self.tree[node_id]['child']:
                #     edge_id = node_id + (action_idx,)
                #     n = self.tree[edge_id]['n']
                #     total_n += n

                for action_index in self.tree[node_id]["child"]:
                    child_id = node_id + (action_index,)
                    n = self.tree[child_id]["n"]
                    q = (self.tree[child_id]["q"] - self.q_min) / (
                        self.q_max - self.q_min
                    )
                    p = self.tree[node_id]["p"][0, action_index]
                    u = (p * np.sqrt(total_n) / (n + 1)) * (
                        self.c1 + np.log((total_n + self.c2 + 1) / self.c2)
                    )
                    UCB_list.append(q + u)

                a_UCB = np.argmax(UCB_list)
                node_id += (a_UCB,)
                node_state, _ = self.d_fn(
                    node_state, a_UCB
                )  # a_UCB를 network의 입력형태로 변환 필요
            else:
                break

        return node_id, node_state

    def expansion(self, leaf_id, leaf_state):
        for action_idx in range(self.action_size):
            child_id = leaf_id + (action_idx,)

            s_child, r_child = self.d_fn(
                leaf_state, action_idx
            )  # action_idx를 network의 입력형태로 변환 필요
            # r_child를 scalar 형태로 변환 -> 네트워크에서 구현?

            p_child, _ = self.p_fn(s_child)

            self.tree[child_id] = {
                "child": [],
                "n": 0.0,
                "q": 0.0,
                "p": p_child,
                "r": r_child_scalar,
            }

            self.tree[leaf_id]["child"].append(action_idx)

        _, leaf_v = self.p_fn(leaf_state)
        # v를 scalar 형태로 변환 -> 네트워크에서 구현?

        return leaf_v

    def backup(self, leaf_id, leaf_v):
        node_id = leaf_id
        node_v = leaf_v
        reward_list = [self.tree[node_id]["r"]]

        while True:
            # Calculate G
            discount_sum_r = 0
            n = len(reward_list) - 1

            for i in range(len(reward_list)):
                discount_sum_r += (self.gamma ** (n - i)) * reward_list[i]

            G = discount_sum_r + ((self.gamma ** (n + 1)) * value)

            # Update Q and N
            q = (self.tree[node_id]["n"] * self.tree[node_id]["q"] + G) / (
                self.tree[node_id]["n"] + 1
            )
            self.tree[node_id]["q"] = q
            self.tree[node_id]["n"] += 1

            # Update max q and min q
            self.q_max = max(q, self.q_max)
            self.q_min = min(q, self.q_min)

            node_id = node_id[:-1]

            if node_id == ():
                break

            reward_list.append(self.tree[node_id]["r"])

    def init_mcts(self, root_state):
        tree = {}
        root_id = (0,)

        p_root, _ = self.p_fn(root_state)

        # init root node
        tree[root_id] = {"child": [], "n": 0.0, "q": 0.0, "p": p_root, "r": 0.0}

        return tree

    def select_root_action(self):
        child = self.tree[self.root_id]["child"]

        n_list = []

        for child_num in child:
            child_idx = self.root_id + (child_num,)
            n_list.append(self.tree[child_idx]["n"])

        pi = np.asarray(n_list) / np.sum(n_list)
        pi_temp = (np.asarray(n_list) ** (1 / self.temp_param)) / (
            np.sum(n_list) ** (1 / self.temp_param)
        )

        noise_probs = self.alpha * np.random.dirichlet(np.ones(self.action_size))
        pi_noise = pi_temp + noise_probs
        pi_noise = pi_noise / np.sum(pi_noise)

        action_idx = np.random.choice(self.action_size, p=pi_noise)

        return action_idx, pi

    def backup(self):
        pass


class Trajectory:
    def __init__(self, state):
        self.states = [state / 255]
        self.actions = [np.zeros((1, 1), dtype=int)]
        self.rewards = [np.zeros((1, 1))]
        self.values = []
        self.policies = []

    def append(self, transition):
        self.states.append(transition["state"] / 255)
        self.actions.append(transition["action"])
        self.rewards.append(transition["reward"])
        self.values.append(transition["value"])
        self.policies.append(transition["pi"])

    def get_bootstrap_value(self, start_idx, td_step, gamma):
        bootstrap_idx = start_idx + td_step
        bootstrap_value = (
            self.values[bootstrap_idx] * (gamma**td_step)
            if bootstrap_idx < len(self.values)
            else np.zeros((1, 1))
        )
        for i, reward in enumerate(self.rewards[start_idx + 1 : bootstrap_idx + 1]):
            bootstrap_value += reward * (gamma**i)

        return bootstrap_value

    def get_stacked_data(self, cur_idx, num_stack):
        # f_dim, r_shape = self.states[cur_idx].shape[1], self.states[cur_idx].shape[2:]
        # num_stack = (f_dim + 1) * num_stack + f_dim
        shape = self.states[cur_idx].shape[-2:]
        actions = np.zeros((num_stack, *shape))
        states = np.zeros((num_stack + 1, *shape))
        states[0] = self.states[cur_idx]

        for i, state in enumerate(self.states[max(0, cur_idx - num_stack) : cur_idx]):
            states[i + 1] = state
            actions[i] = np.ones(shape) * self.actions[i]

        return states, actions
