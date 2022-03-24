from collections import defaultdict
from collections.abc import Iterable
import os
import torch
import numpy as np

torch.backends.cudnn.benchmark = True

from core.network import Network
from core.optimizer import Optimizer

from core.buffer import ReplayBuffer
from core.buffer import PERBuffer
from .base import BaseAgent


class Muzero(BaseAgent):
    action_type = "discrete"
    """MuZero agent.

    Args:
        -
    """

    def __init__(
        self,
        # MuZero
        state_size,
        action_size,
        network="muzero_resnet",
        head="residualblock",
        hidden_size=256,
        gamma=0.997,
        batch_size=16,
        start_train_step=0,
        trajectory_size=200,
        value_loss_weight=0.25,
        num_simulation=50,
        num_unroll=5,
        num_td_step=10,
        num_support=300,
        num_stack=32,
        buffer_size=125000,
        device=None,
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

        if isinstance(state_size, Iterable):
            self.trajectory_type = Trajectory
            stack_dim = (state_size[0] + 1) * num_stack + state_size[0]
        else:
            self.trajectory_type = TrajectoryGym
            stack_dim = (state_size + 1) * num_stack + state_size

        self.network = Network(
            network,
            state_size,
            action_size,
            stack_dim,
            num_support,
            D_hidden=hidden_size,
            head=head,
        ).to(self.device)

        self.target_network = Network(
            network,
            state_size,
            action_size,
            stack_dim,
            num_support,
            D_hidden=hidden_size,
            head=head,
        ).to("cpu")
        self.target_network.load_state_dict(self.network.state_dict())

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
        # self.learn_period = learn_period
        # self.learn_period_stamp = 0
        # self.buffer_size = buffer_size
        # self.uniform_sample_prob = uniform_sample_prob
        # self.beta_add = (1 - beta) / run_step
        # self.memory = PERBuffer(self.buffer_size, uniform_sample_prob)
        # no priority
        self.buffer_size = buffer_size
        self.memory = ReplayBuffer(buffer_size)
        self.memory.first_store = False

        # MCTS
        self.mcts = MCTS(
            self.target_network,
            self.action_size,
            self.num_simulation,
            self.num_unroll,
            self.gamma,
        )

    @torch.no_grad()
    def act(self, state, training=True):
        self.target_network.train(training)
        # TODO: if eval

        if not self.trajectory:
            self.trajectory = self.trajectory_type(state)
            self.update_target()

        states, actions = self.trajectory.get_stacked_data(
            self.trajectory_step_stamp, self.num_stack
        )
        self.device, swap = "cpu", self.device
        states = self.as_tensor(np.expand_dims(states, axis=0))
        actions = self.as_tensor(np.expand_dims(actions, axis=0))
        self.device = swap
        root_state = self.target_network.representation(states, actions)
        action, pi, value = self.mcts.run_mcts(root_state)
        action = np.array(((action,),))

        return {"action": action, "value": value, "pi": pi}

    def learn(self):
        trajectories = self.memory.sample(self.batch_size)

        transitions = defaultdict(list)
        for i, trajectory in enumerate(trajectories["trajectory"]):
            trajectory_len = len(trajectory["values"])
            start_idx = np.random.choice(trajectory_len)

            # make target
            last_idx = start_idx + self.num_unroll + 1
            actions = trajectory["actions"][start_idx:last_idx]
            policies = trajectory["policies"][start_idx:last_idx]
            rewards = trajectory["rewards"][start_idx:last_idx]
            values = [
                trajectory.get_bootstrap_value(i, self.num_td_step, self.gamma)
                for i in range(start_idx, last_idx)
            ]
            for _ in range(trajectory_len, last_idx):
                rewards.append(np.zeros((1, 1)))
                actions.append(np.random.choice(self.action_size, size=(1, 1)))
                policies.append(np.ones(self.action_size) / self.action_size)

            s, a = trajectory.get_stacked_data(start_idx, self.num_stack)
            transitions["stacked_state"].append(s)
            transitions["stacked_action"].append(a)
            transitions["action"].append(actions)
            transitions["reward"].append(rewards)
            transitions["policy"].append(policies)
            transitions["value"].append(values)
            transitions["gradient_scale"].append(
                np.ones(self.num_unroll + 1)
                * min(self.num_unroll, trajectory_len + 1 - start_idx)
            )

        for key in transitions.keys():
            value = np.stack(transitions[key], axis=0)
            if value.shape[-1] == 1:
                value = value.squeeze(axis=-1)
            transitions[key] = self.as_tensor(value)

        stacked_state = transitions["stacked_state"]
        stacked_action = transitions["stacked_action"]
        target_action = transitions["action"]
        target_reward = transitions["reward"]
        target_policy = transitions["policy"]
        target_value = transitions["value"]
        gradient_scale = transitions["gradient_scale"]

        target_reward = self.network.converter.scalar2vector(target_reward)
        target_value = self.network.converter.scalar2vector(target_value)

        # stacked_state batch, 32*3+1, 96,96
        hidden_state = self.network.representation(stacked_state, stacked_action)
        pi, value = self.network.prediction(hidden_state)
        reward = torch.zeros(target_reward.shape, device=self.device)
        prediction = [(value, reward, pi)]

        for i in range(1, self.num_unroll + 1):
            hidden_state, reward = self.network.dynamics(
                hidden_state, target_action[:, i]
            )
            pi, value = self.network.prediction(hidden_state)
            hidden_state.register_hook(lambda x: x * 0.5)
            prediction.append((value, reward, pi))

        # compute start step
        value, reward, pi = prediction[0]
        policy_loss = (-target_policy[:, 0] * torch.nn.LogSoftmax(dim=1)(pi)).sum(1)
        value_loss = (-target_value[:, 0] * torch.log(value + 1e-6)).sum(1)
        reward_loss = torch.zeros_like(value_loss, device=self.device)

        # comput remain step
        for i, (value, reward, pi) in enumerate(prediction[1:], start=1):
            local_loss = (-target_policy[:, i] * torch.log(pi + 1e-6)).sum(1)
            local_loss.register_hook(lambda x: x / gradient_scale[:, i])
            policy_loss += local_loss
            value_loss += (-target_value[:, i] * torch.log(value + 1e-6)).sum(1)
            local_loss.register_hook(lambda x: x / gradient_scale[:, i])
            value_loss += local_loss
            reward_loss += (-target_reward[:, i] * torch.log(reward + 1e-6)).sum(1)
            local_loss.register_hook(lambda x: x / gradient_scale[:, i])
            reward_loss += local_loss

        loss = (self.value_loss_weight * value_loss + policy_loss + reward_loss).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.num_learn += 1

        result = {
            "loss": loss.item(),
            "policy_loss": policy_loss.mean().item(),
            "value_loss": value_loss.mean().item(),
            "reward_loss": reward_loss.mean().item(),
        }
        return result

    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def process(self, transitions, step):
        result = {}

        # Process per step
        self.memory.store(transitions)
        self.time_t = step

        if (
            self.memory.buffer_counter >= self.batch_size
            and self.time_t >= self.start_train_step
        ):
            result = self.learn()
            self.learning_rate_decay(step)

        return result

    def get_temperature(self, step):
        if step < self.run_step * 0.5:
            t = 1.0
        elif step < self.run_step * 0.75:
            t = 0.5
        else:
            t = 0.25
        return t

    def interact_callback(self, transition):
        _transition = None
        self.trajectory_step_stamp += 1

        self.trajectory["states"].append(transition["state"])
        self.trajectory["actions"].append(transition["action"])
        self.trajectory["rewards"].append(transition["reward"])
        self.trajectory["values"].append(transition["value"])
        self.trajectory["policies"].append(transition["pi"])

        if transition["done"] or self.trajectory_step_stamp >= self.trajectory_size:
            self.trajectory_step_stamp = 0  # never self.trajectory_step_stamp -= period
            # TODO: if not terminal -> n-step calc
            self.trajectory["values"].append(np.zeros((1, 1)))
            self.trajectory["policies"].append(
                np.ones(self.action_size) / self.action_size
            )

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


class MCTS:
    def __init__(self, network, action_size, n_mcts, n_unroll, gamma):
        self.network = network
        self.p_fn = network.prediction  # prediction function
        self.d_fn = network.dynamics  # dynamics function

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

    @torch.no_grad()
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

    @torch.no_grad()
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
                    UCB_list.append((q + u).cpu())

                a_UCB = np.argmax(UCB_list)
                node_id += (a_UCB,)
                node_state, _ = self.d_fn(node_state, torch.FloatTensor([a_UCB]))
            else:
                break

        return node_id, node_state

    @torch.no_grad()
    def expansion(self, leaf_id, leaf_state):
        for action_idx in range(self.action_size):
            child_id = leaf_id + (action_idx,)

            action = np.ones((1, 1)) * action_idx
            s_child, r_child = self.d_fn(leaf_state, torch.FloatTensor(action))

            # r_child를 scalar 형태로 변환 -> 네트워크에서 구현?
            r_child_scalar = self.network.converter.vector2scalar(r_child).item()

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
        leaf_v = self.network.converter.vector2scalar(leaf_v).item()

        return leaf_v

    @torch.no_grad()
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

            G = discount_sum_r + ((self.gamma ** (n + 1)) * node_v)

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

    @torch.no_grad()
    def init_mcts(self, root_state):
        tree = {}
        root_id = (0,)

        p_root, _ = self.p_fn(root_state)

        # init root node
        tree[root_id] = {"child": [], "n": 0.0, "q": 0.0, "p": p_root, "r": 0.0}

        return tree

    @torch.no_grad()
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


class Trajectory(dict):
    def __init__(self, state):
        self["states"] = [state]
        self["actions"] = [np.zeros((1, 1), dtype=int)]
        self["rewards"] = [np.zeros((1, 1))]
        self["values"] = []
        self["policies"] = []
        self["priorities"] = None
        self["priority"] = None

    def get_bootstrap_value(self, start, num_td_step, gamma):
        last = start + num_td_step
        value = self["values"][last] if last < len(self["values"]) else np.zeros((1, 1))
        for reward in self["rewards"][start + 1 : last + 1]:
            value = reward + gamma * value
        return value

    def get_stacked_data(self, cur_idx, num_stack):
        shape = self["states"][cur_idx].shape[-2:]
        actions = np.zeros((num_stack, *shape))
        states = np.zeros((num_stack + 1, *shape))
        states[0] = self["states"][cur_idx]

        for i, state in enumerate(
            self["states"][max(0, cur_idx - num_stack) : cur_idx]
        ):
            states[i + 1] = state
            actions[i] = np.ones(shape) * self["actions"][i]

        return states, actions


class TrajectoryGym(Trajectory):
    def get_stacked_data(self, cur_idx, num_stack):
        d_channel = self["states"][cur_idx].shape[1]
        actions = np.zeros(num_stack)
        states = np.zeros((num_stack + 1) * d_channel)
        states[0:d_channel] = self["states"][cur_idx]

        for i, state in enumerate(
            self["states"][max(0, cur_idx - num_stack) : cur_idx]
        ):
            si = (i + 1) * d_channel
            states[si : si + d_channel] = state
            actions[i] = np.ones(1) * self["actions"][i]

        return states, actions
