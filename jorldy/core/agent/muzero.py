import os
import torch
import numpy as np
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

from core.network import Network
from core.optimizer import Optimizer
from collections import defaultdict
from collections.abc import Iterable

from core.buffer import MuzeroPERBuffer
from .base import BaseAgent


class Muzero(BaseAgent):
    action_type = "discrete"
    """MuZero agent.

    Args:
        -
        num_rb: the number of residual block.
        lr_decay: lr_decay option which apply decayed weight on parameters of network.
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
        start_train_step=2000,
        policy_train_delay=2000,
        max_trajectory_size=200,
        value_loss_weight=1.0,
        num_unroll=5,
        num_td_step=10,
        num_support=300,
        num_stack=32,
        num_rb=16,
        buffer_size=125000,
        device=None,
        run_step=1e6,
        num_workers=1,
        # Optim
        lr_decay=True,
        optim_config={
            "name": "adam",
            "weight_decay": 1e-4,
            "lr": 5e-4,
        },
        # PER
        alpha=1.0,
        beta=1.0,
        learn_period=1,
        uniform_sample_prob=1e-3,
        # MCTS
        num_mcts=50,
        num_eval_mcts=15,
        mcts_alpha_max=0.5,
        mcts_alpha_min=0.0,
        # Optional Feature
        use_prev_rand_action=True,
        use_over_rand_action=True,
        use_uniform_policy=True,
        use_ssc_loss=False,
        **kwargs,
    ):
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.channel = state_size[0] if isinstance(state_size, Iterable) else state_size

        self.network = Network(
            network,
            state_size,
            action_size,
            num_stack,
            num_support,
            num_rb=num_rb,
            D_hidden=hidden_size,
            head=head,
        ).to(self.device)

        self.target_network = Network(
            network,
            state_size,
            action_size,
            num_stack,
            num_support,
            num_rb=num_rb,
            D_hidden=hidden_size,
            head=head,
        ).to("cpu")
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = Optimizer(**optim_config, params=self.network.parameters())

        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.start_train_step = start_train_step
        self.policy_train_delay = (
            start_train_step + policy_train_delay if policy_train_delay > 0 else 0
        )
        self.value_loss_weight = value_loss_weight

        self.max_trajectory_size = max_trajectory_size
        self.num_unroll = num_unroll
        self.num_td_step = num_td_step
        self.num_stack = num_stack

        self.time_t = 0
        self.trajectory_step_stamp = 0
        self.run_step = run_step
        self.lr_decay = lr_decay
        self.num_workers = num_workers
        self.num_learn = 0
        self.num_transitions = 0

        self.trajectory = None

        self.use_over_rand_action = use_over_rand_action
        self.use_prev_rand_action = use_prev_rand_action
        self.use_uniform_policy = use_uniform_policy

        # PER
        self.alpha = alpha
        self.beta = beta
        self.learn_period = learn_period
        self.learn_period_stamp = 0
        self.buffer_size = buffer_size
        self.uniform_sample_prob = uniform_sample_prob
        self.beta_add = (1 - beta) / run_step
        self.memory = MuzeroPERBuffer(self.buffer_size, uniform_sample_prob)

        # MCTS
        self.num_mcts = num_mcts
        self.num_eval_mcts = num_eval_mcts
        self.mcts_alpha_max = mcts_alpha_max
        self.mcts_alpha_min = mcts_alpha_min

        self.mcts = MCTS(
            self.target_network,
            self.action_size,
            self.num_unroll,
            self.gamma,
            self.policy_train_delay > 0,
        )

        # Self Supervised Consistency Loss
        self.use_ssc_loss = use_ssc_loss

    @torch.no_grad()
    def act(self, state, training=True):
        self.target_network.eval()
        
        if not self.trajectory:
            self.init_trajectory(state)
            self.update_target()
            
        stacked_s, stacked_a = self.get_stacked_data(
            self.trajectory, self.trajectory_step_stamp, self.num_stack
        )

        self.device, swap = "cpu", self.device
        stacked_s = self.as_tensor(np.expand_dims(stacked_s, axis=0))
        stacked_a = self.as_tensor(np.expand_dims(stacked_a, axis=0))
        self.device = swap
        root_state = self.target_network.representation(stacked_s, stacked_a)

        if training:
            n_mcts = self.num_mcts
        else:
            self.mcts.use_uniform_policy = False
            n_mcts = self.num_eval_mcts
        action, pi, value = self.mcts.run_mcts(root_state, n_mcts, training)
        action = np.array(action if training else np.argmax(pi), ndmin=2)

        return {"action": action, "value": np.array(value), "pi": pi}

    def learn(self):
        transitions, weights, indices, sampled_p, mean_p = self.memory.sample(
            self.beta, self.batch_size
        )

        _transitions = defaultdict(list)
        for trajectory, start in transitions:
            trajectory_len = len(trajectory["values"])

            # make target
            end = start + self.num_unroll + 1
            stack_len = self.num_stack + self.num_unroll
            state, action = self.get_stacked_data(trajectory, end - 1, stack_len)

            policy = trajectory["policies"][start:end]
            reward = trajectory["rewards"][start:end]
            value = [self.get_bootstrap_value(trajectory, i) for i in range(start, end)]

            for i in reversed(range(end - trajectory_len)):
                reward.append(np.zeros((1, 1)))
                policy.append(
                    np.full(self.action_size, 1 / self.action_size)
                    if self.use_uniform_policy
                    else np.zeros(self.action_size)
                )
                if self.use_over_rand_action:
                    action[stack_len - i - 1] = np.random.randint(
                        self.action_size, size=(1, 1)
                    )

            _transitions["state"].append(state)
            _transitions["action"].append(action)
            _transitions["reward"].append(reward)
            _transitions["policy"].append(policy)
            _transitions["value"].append(value)

        for key in _transitions.keys():
            value = np.stack(_transitions[key], axis=0)
            if value.shape[-1] == 1:
                value = value.squeeze(axis=-1)
            _transitions[key] = self.as_tensor(value)

        state = _transitions["state"]
        action = _transitions["action"]
        selected_action = action[:, -self.num_unroll :]
        target_policy = _transitions["policy"]
        target_reward_s = _transitions["reward"]
        target_value_s = _transitions["value"]

        target_reward = self.network.converter.scalar2vector(target_reward_s)
        target_value = self.network.converter.scalar2vector(target_value_s)

        stack_s, stack_a = (
            state[:, : self.channel * (self.num_stack + 1)],
            action[:, : self.num_stack],
        )

        # comput start step loss
        hidden_state = self.network.representation(stack_s, stack_a)
        pi, value = self.network.prediction(hidden_state)

        value_s = self.network.converter.vector2scalar(torch.exp(value))
        max_V = torch.max(value_s).item()
        min_V = torch.min(value_s).item()
        max_R = float("-inf")
        min_R = float("inf")

        # Update sum tree
        td_error = abs(value_s - target_value_s[:, 0])
        p_j = torch.pow(td_error, self.alpha)
        for i, p in zip(indices, p_j):
            self.memory.update_priority(p.item(), i)

        policy_loss = -(target_policy[:, 0] * pi).sum(1)
        value_loss = -(target_value[:, 0] * value).sum(1)
        reward_loss = torch.zeros(self.batch_size, device=self.device)
        ssc_loss = torch.zeros(self.batch_size, device=self.device)
        
        # comput unroll step loss
        for end, i in enumerate(range(1, self.num_unroll + 1), self.num_stack + 1):
            stack_s, stack_a = (
                state[:, self.channel * i : self.channel * (end + 1)],
                action[:, i:end],
            )

            hidden_state, reward = self.network.dynamics(
                hidden_state, selected_action[:, i - 1 : i]
            )

            if self.use_ssc_loss:
                with torch.no_grad():
                    y = self.network.representation(stack_s, stack_a)
                ssc_loss -= F.cosine_similarity(y.flatten(1), hidden_state.flatten(1))

            pi, value = self.network.prediction(hidden_state)
            hidden_state.register_hook(lambda x: x * 0.5)

            policy_loss += -(target_policy[:, i] * pi).sum(1)
            value_loss += -(target_value[:, i] * value).sum(1)
            reward_loss += -(target_reward[:, i] * reward).sum(1)

            reward_s = self.network.converter.vector2scalar(torch.exp(reward))
            value_s = self.network.converter.vector2scalar(torch.exp(value))
            
            max_V = max(max_V, torch.max(value_s).item())
            min_V = min(min_V, torch.min(value_s).item())
            max_R = max(max_R, torch.max(reward_s).item())
            min_R = min(min_R, torch.min(reward_s).item())

        weights = torch.unsqueeze(torch.FloatTensor(weights).to(self.device), -1)
        loss = self.value_loss_weight * value_loss + policy_loss + reward_loss
        weighted_loss = (weights * (loss.mean(-1) + ssc_loss)).mean()

        gradient_scale = 1 / self.num_unroll
        weighted_loss.register_hook(lambda x: x * gradient_scale)

        self.optimizer.zero_grad(set_to_none=True)
        weighted_loss.backward()
        self.optimizer.step()
        self.num_learn += 1

        result = {
            "loss": loss.mean().item(),
            "weighted_loss": weighted_loss.item(),
            "P_loss": policy_loss.mean().item(),
            "V_loss": value_loss.mean().item(),
            "R_loss": reward_loss.mean().item(),
            "SSC_loss": ssc_loss.mean().item(),
            "max_V": max_V,
            "min_V": min_V,
            "max_R": max_R,
            "min_R": min_R,
            "sampled_p": sampled_p,
            "mean_p": mean_p,
            "num_learn": self.num_learn,
            "num_transitions": self.num_transitions,
        }
        return result

    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def process(self, transitions, step):
        result = {}
        self.num_transitions += len(transitions)

        # Process per step
        delta_t = step - self.time_t
        self.memory.store(transitions)
        self.time_t = step
        self.learn_period_stamp += delta_t

        # Annealing beta
        self.beta = min(1.0, self.beta + (self.beta_add * delta_t))

        if (
            self.learn_period_stamp >= self.learn_period
            and self.memory.buffer_counter >= self.batch_size
            and self.time_t >= self.start_train_step
        ):
            result = self.learn()
            if self.lr_decay:
                self.learning_rate_decay(step)
            self.set_temperature(step)
            self.learn_period_stamp -= self.learn_period

        return result

    def interact_callback(self, transition):
        _transition = None
        self.trajectory_step_stamp += 1

        self.trajectory["states"].append(transition["next_state"])
        self.trajectory["actions"].append(transition["action"])
        self.trajectory["rewards"].append(transition["reward"])
        self.trajectory["values"].append(transition["value"])
        self.trajectory["policies"].append(transition["pi"])

        if transition["done"] or self.trajectory_step_stamp >= self.max_trajectory_size:
            self.trajectory_step_stamp = 0  # never self.trajectory_step_stamp -= period

            # TODO: if not terminal -> n-step calc
            self.trajectory["values"].append(np.zeros((1, 1)))
            self.trajectory["policies"].append(
                np.full(self.action_size, 1 / self.action_size)
                if self.use_uniform_policy
                else np.zeros(self.action_size)
            )
            priorities = np.zeros(len(self.trajectory["values"]) - 1)
            for i, v in enumerate(self.trajectory["values"][:-1]):
                z = self.get_bootstrap_value(self.trajectory, i)
                priorities[i] = abs(v - z) ** self.alpha

            _transition = {"trajectory": self.trajectory, "priorities": priorities}
            self.trajectory = None

        return _transition

    def init_trajectory(self, state):
        self.trajectory = {
            "states": [state],
            "actions": [np.zeros((1, 1), dtype=int)],
            "rewards": [np.zeros((1, 1))],
            "values": [],
            "policies": [],
        }

    def get_bootstrap_value(self, trajectory, start):
        end = start + self.num_td_step
        values = trajectory["values"]
        value = values[end] if end < len(values) else np.zeros((1, 1))

        for reward in reversed(trajectory["rewards"][start + 1 : end + 1]):
            value = reward + self.gamma * value

        return value

    def get_stacked_data(self, trajectory, cur_idx, num_stack):
        cut = max(0, num_stack - cur_idx)
        start = max(0, cur_idx - num_stack)
        end = min(len(trajectory["states"]) - 1, cur_idx)
        shape = trajectory["states"][start].shape[1:]
        stacked_a = np.zeros(num_stack, int)
        stacked_s = np.zeros((num_stack + 1, *shape), np.float32)

        if self.use_prev_rand_action:
            stacked_a[:cut] = np.random.randint(self.action_size, size=cut)

        for n, i in enumerate(range(start, end), start=cut):
            stacked_a[n] = trajectory["actions"][i + 1]
            stacked_s[n] = trajectory["states"][i]

        stacked_s[cut + end - start] = trajectory["states"][end]
        stacked_s = stacked_s.reshape(((num_stack + 1) * shape[0], *shape[1:]))
        stacked_a = stacked_a.reshape((num_stack))

        return stacked_s, stacked_a

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

    def set_distributed(self, id):
        assert self.num_workers > 1
        self.mcts.alpha = self.mcts_alpha_min + id * (
            self.mcts_alpha_max - self.mcts_alpha_min
        ) / (self.num_workers - 1)
        
        # self.mcts.c_ucb = 1 + id * (1/(self.num_workers-1))

        return self

    def set_temperature(self, step):
        if step < self.run_step * 0.5:
            self.mcts.temp_param = 1.0
        elif step < self.run_step * 0.75:
            self.mcts.temp_param = 0.5
        else:
            self.mcts.temp_param = 0.25

    def sync_in(self, weights, temperature, use_uniform_policy):
        self.network.load_state_dict(weights)
        self.mcts.temp_param = temperature
        self.mcts.use_uniform_policy = use_uniform_policy

    def sync_out(self, device="cpu"):
        weights = self.network.state_dict()
        for k, v in weights.items():
            weights[k] = v.to(device)
        sync_item = {
            "weights": weights,
            "temperature": self.mcts.temp_param,
            "use_uniform_policy": self.policy_train_delay > self.time_t,
        }
        return sync_item


class MCTS:
    def __init__(self, network, action_size, n_unroll, gamma, use_uniform_policy):
        self.network = network
        self.p_fn = network.prediction  # prediction function
        self.d_fn = network.dynamics  # dynamics function

        self.use_uniform_policy = use_uniform_policy
        self.action_size = action_size
        self.n_unroll = n_unroll + 1

        self.gamma = gamma
        self.temp_param = 1.0

        self.c1 = 1.25
        self.c2 = 19625
        self.alpha = 0.3
        
        self.c_ucb = 1.0

        self.q_min = 0
        self.q_max = 0

        self.root_id = (0,)
        self.tree = {}

    @torch.no_grad()
    def run_mcts(self, root_state, num_mcts, training):
        self.tree = self.init_mcts(root_state, training)

        for i in range(num_mcts):
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
                
                for action_index in self.tree[node_id]["child"]:
                    child_id = node_id + (action_index,)
                    n = self.tree[child_id]["n"]
                    q = (
                        self.tree[child_id]["q"]
                        if n > 0
                        else self.tree[child_id]["v"] * 0.8
                    )
                    q = (q - self.q_min) / (self.q_max - self.q_min)
                    p = self.tree[node_id]["p"][0, action_index]
                    u = (p * np.sqrt(total_n) / (n + 1)) * (
                        self.c1 + np.log((total_n + self.c2 + 1) / self.c2)
                    )
                    UCB_list.append((q + self.c_ucb*u).cpu())
                    
                    # UCB_list.append((q + u).cpu())
                
                max_UCB = np.max(UCB_list)
                max_list = [a for a, v in enumerate(UCB_list) if v == max_UCB]
                a_UCB = np.random.choice(max_list)
                
                node_id += (a_UCB,)
                node_state = self.tree[node_id]["s"]
            else:
                break

        return node_id, node_state

    @torch.no_grad()
    def expansion(self, leaf_id, leaf_state):
        repeat_size = [1] * len(leaf_state.shape)
        repeat_size[0] = self.action_size

        leaf_state_repeat = leaf_state.repeat(repeat_size)
        action_child = torch.arange(0, self.action_size).unsqueeze(1)

        s_child, r_child = self.d_fn(leaf_state_repeat, action_child)
        r_child = torch.exp(r_child)
        r_child_scalar = self.network.converter.vector2scalar(r_child)

        p_child, v_child = self.p_fn(s_child)
        p_child = (
            torch.full((self.action_size, self.action_size), 1 / self.action_size)
            if self.use_uniform_policy
            else torch.exp(p_child)
        )
        v_child = torch.exp(v_child)
        v_child_scalar = self.network.converter.vector2scalar(v_child)

        for action_idx in range(self.action_size):
            child_id = leaf_id + (action_idx,)

            self.tree[child_id] = {
                "child": [],
                "s": s_child[action_idx].unsqueeze(0),
                "n": 0.0,
                "q": 0.0,
                "p": p_child[action_idx].unsqueeze(0),
                "v": v_child_scalar[action_idx].item(),
                "r": r_child_scalar[action_idx].item(),
            }

            self.tree[leaf_id]["child"].append(action_idx)

        leaf_v = self.tree[leaf_id]["v"]

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
            # if self.tree[node_id]["n"] == 0:
            #     self.tree[node_id]["q"] = 0
            
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
    def init_mcts(self, root_state, training):
        tree = {}
        root_id = (0,)

        p_root, v_root = self.p_fn(root_state)
        # p_root = (
        #     torch.full((1, self.action_size), 1 / self.action_size)
        #     if self.use_uniform_policy
        #     else torch.exp(p_root)
        # )
        
        
        if self.use_uniform_policy:
            p_root = torch.full((1, self.action_size), 1 / self.action_size)
        else:
            p_root = torch.exp(p_root)
            
            if training:
                noise_probs = np.random.dirichlet(self.alpha * np.ones(self.action_size))
                p_root = p_root * 0.8 + noise_probs * 0.2
                p_root = p_root / torch.sum(p_root)
            
        v_root = torch.exp(v_root)
        v_root_scalar = self.network.converter.vector2scalar(v_root).item()

        # init root node
        tree[root_id] = {
            "child": [],
            "s": root_state,
            "n": 0.0,
            "q": 0.0,
            "p": p_root,
            "v": v_root_scalar,
            "r": 0.0,
        }

        return tree

    @torch.no_grad()
    def select_root_action(self):
        child = self.tree[self.root_id]["child"]

        n_list = []

        for child_num in child:
            child_idx = self.root_id + (child_num,)
            n_list.append(self.tree[child_idx]["n"])

        pi = np.asarray(n_list) / np.sum(n_list)
        pi_temp = np.asarray(n_list) ** (1 / self.temp_param)
        pi_temp = pi_temp / np.sum(pi_temp)

        noise_probs = np.random.dirichlet(self.alpha * np.ones(self.action_size))
        pi_noise = pi_temp * 0.8 + noise_probs * 0.2
        pi_noise = pi_noise / np.sum(pi_noise)

        action_idx = np.random.choice(self.action_size, p=pi_noise)

        return action_idx, pi
