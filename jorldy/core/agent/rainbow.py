from collections import deque
import torch

torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
import numpy as np

from core.network import Network
from core.optimizer import Optimizer
from core.buffer import PERBuffer
from .dqn import DQN


class Rainbow(DQN):
    """Rainbow agent.

    Args:
        state_size (int): dimension of state.
        action_size (int): dimension of action.
        hidden_size (int): dimension of hidden unit.
        network (str): key of network class in _network_dict.txt.
        head (str): key of head in _head_dict.txt.
        optim_config (dict): dictionary of the optimizer info.
            (key: 'name', value: name of optimizer)
        gamma (float): discount factor.
        buffer_size (int): the size of the memory buffer.
        batch_size (int): the number of samples in the one batch.
        start_train_step (int): steps to start learning.
        target_update_period (int): period to update the target network. (unit: step)
        run_step (int): the number of total steps.
        lr_decay: lr_decay option which apply decayed weight on parameters of network.
        n_step: number of steps in multi-step Q learning.
        alpha (float): prioritization exponent.
        beta (float): initial value of degree to use importance sampling.
        learn_period (int): period to train (unit: step)
        uniform_sample_prob (float): ratio of uniform random sampling.
        noise_type (str): NoisyNet noise type. One of ['factorized', 'independent']
            ('factorized': Factorized Gaussian Noise, else: Independent Gaussian Noise)
        v_min (float): minimum value of support.
        v_max (float): maximum value of support.
        num_support (int): number of support.
        device (str): device to use.
            (e.g. 'cpu' or 'gpu'. None can also be used, and in this case, the cpu is used.)
    """

    def __init__(
        self,
        state_size,
        action_size,
        hidden_size=512,
        network="rainbow",
        head="mlp",
        optim_config={"name": "adam"},
        gamma=0.99,
        buffer_size=50000,
        batch_size=64,
        start_train_step=2000,
        target_update_period=500,
        run_step=1e6,
        lr_decay=True,
        # MultiStep
        n_step=4,
        # PER
        alpha=0.6,
        beta=0.4,
        learn_period=4,
        uniform_sample_prob=1e-3,
        # Noisy
        noise_type="factorized",  # [independent, factorized]
        # C51
        v_min=-10,
        v_max=10,
        num_support=51,
        device=None,
        **kwargs,
    ):
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.action_size = action_size
        self.network = Network(
            network,
            state_size,
            action_size,
            num_support,
            noise_type,
            D_hidden=hidden_size,
            head=head,
        ).to(self.device)
        self.target_network = Network(
            network,
            state_size,
            action_size,
            num_support,
            noise_type,
            D_hidden=hidden_size,
            head=head,
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = Optimizer(**optim_config, params=self.network.parameters())
        self.gamma = gamma
        self.batch_size = batch_size
        self.start_train_step = start_train_step
        self.target_update_stamp = 0
        self.target_update_period = target_update_period
        self.num_learn = 0
        self.time_t = 0
        self.run_step = run_step
        self.lr_decay = lr_decay

        # MultiStep
        self.n_step = n_step
        self.tmp_buffer = deque(maxlen=n_step)

        # PER
        self.alpha = alpha
        self.beta = beta
        self.learn_period = learn_period
        self.learn_period_stamp = 0
        self.uniform_sample_prob = uniform_sample_prob
        self.beta_add = (1 - beta) / run_step

        # C51
        self.v_min = v_min
        self.v_max = v_max
        self.num_support = num_support

        # MultiStep
        self.memory = PERBuffer(buffer_size, uniform_sample_prob)

        # C51
        self.delta_z = (v_max - v_min) / (num_support - 1)
        self.z = torch.linspace(v_min, v_max, num_support, device=self.device).view(
            1, -1
        )

    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training)

        if training and self.memory.size < max(self.batch_size, self.start_train_step):
            batch_size = (
                state[0].shape[0] if isinstance(state, list) else state.shape[0]
            )
            action = np.random.randint(0, self.action_size, size=(batch_size, 1))
        else:
            logits = self.network(self.as_tensor(state), training)
            _, q_action = self.logits2Q(logits)
            action = torch.argmax(q_action, -1, keepdim=True).cpu().numpy()
        return {"action": action}

    def learn(self):
        transitions, weights, indices, sampled_p, mean_p = self.memory.sample(
            self.beta, self.batch_size
        )
        for key in transitions.keys():
            transitions[key] = self.as_tensor(transitions[key])

        state = transitions["state"]
        action = transitions["action"]
        reward = transitions["reward"]
        next_state = transitions["next_state"]
        done = transitions["done"]

        logit = self.network(state, True)
        p_logit, q_action = self.logits2Q(logit)

        action_eye = torch.eye(self.action_size).to(self.device)
        action_onehot = action_eye[action.long()]

        p_action = torch.squeeze(action_onehot @ p_logit, 1)

        target_dist = torch.zeros(
            self.batch_size, self.num_support, device=self.device, requires_grad=False
        )
        with torch.no_grad():
            # Double
            _, next_q_action = self.logits2Q(self.network(next_state, True))

            target_p_logit, _ = self.logits2Q(self.target_network(next_state, True))

            target_action = torch.argmax(next_q_action, -1, keepdim=True)
            target_action_onehot = action_eye[target_action.long()]
            target_p_action = torch.squeeze(target_action_onehot @ target_p_logit, 1)

            Tz = self.z
            for i in reversed(range(self.n_step)):
                Tz = (
                    reward[:, i].expand(-1, self.num_support)
                    + (1 - done[:, i]) * self.gamma * Tz
                )

            b = torch.clamp(Tz - self.v_min, 0, self.v_max - self.v_min) / self.delta_z
            l = torch.floor(b).long()
            u = torch.ceil(b).long()

            support_eye = torch.eye(self.num_support, device=self.device)
            l_support_onehot = support_eye[l]
            u_support_onehot = support_eye[u]

            l_support_binary = torch.unsqueeze(u - b, -1)
            u_support_binary = torch.unsqueeze(b - l, -1)
            target_p_action_binary = torch.unsqueeze(target_p_action, -1)

            lluu = (
                l_support_onehot * l_support_binary
                + u_support_onehot * u_support_binary
            )

            target_dist += done[:, 0, :] * torch.mean(
                l_support_onehot * u_support_onehot + lluu, 1
            )
            target_dist += (1 - done[:, 0, :]) * torch.sum(
                target_p_action_binary * lluu, 1
            )
            target_dist /= torch.clamp(
                torch.sum(target_dist, 1, keepdim=True), min=1e-8
            )

        max_Q = torch.max(q_action).item()
        max_logit = torch.max(logit).item()
        min_logit = torch.min(logit).item()

        # PER
        KL = -(target_dist * torch.clamp(p_action, min=1e-8).log()).sum(-1)
        p_j = torch.pow(KL, self.alpha)

        for i, p in zip(indices, p_j):
            self.memory.update_priority(p.item(), i)

        weights = torch.unsqueeze(torch.FloatTensor(weights).to(self.device), -1)

        loss = (weights * KL).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.num_learn += 1

        result = {
            "loss": loss.item(),
            "beta": self.beta,
            "max_Q": max_Q,
            "max_logit": max_logit,
            "min_logit": min_logit,
            "sampled_p": sampled_p,
            "mean_p": mean_p,
        }

        return result

    def process(self, transitions, step):
        result = {}

        # Process per step
        delta_t = step - self.time_t
        self.memory.store(transitions)
        self.time_t = step
        self.target_update_stamp += delta_t
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
            self.learn_period_stamp -= self.learn_period

        # Process per step if train start
        if self.num_learn > 0 and self.target_update_stamp >= self.target_update_period:
            self.update_target()
            self.target_update_stamp -= self.target_update_period

        return result

    def logits2Q(self, logits):
        _logits = logits.view(logits.shape[0], self.action_size, self.num_support)
        p_logit = torch.exp(F.log_softmax(_logits, dim=-1))

        z_action = self.z.expand(p_logit.shape[0], self.action_size, self.num_support)
        q_action = torch.sum(z_action * p_logit, dim=-1)

        return p_logit, q_action

    def interact_callback(self, transition):
        _transition = {}
        self.tmp_buffer.append(transition)
        if len(self.tmp_buffer) == self.n_step:
            _transition["state"] = self.tmp_buffer[0]["state"]
            _transition["action"] = self.tmp_buffer[0]["action"]
            _transition["next_state"] = self.tmp_buffer[-1]["next_state"]

            for key in self.tmp_buffer[0].keys():
                if key not in ["state", "action", "next_state"]:
                    _transition[key] = np.stack(
                        [t[key] for t in self.tmp_buffer], axis=1
                    )

        return _transition
