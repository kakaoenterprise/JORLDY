from collections import deque
import torch

torch.backends.cudnn.benchmark = True
import numpy as np

from core.buffer import PERBuffer
from .dqn import DQN


class ApeX(DQN):
    """Ape-X agent.

    Args:
        epsilon (float): epsilon in epsilon_i greedy policy (each i-th actor)
            where epsilon_i=epsilon^(1+(i/(N-1))*alpha).
        epsilon_alpha (float): alpha in epsilon_i greedy policy
            where epsilon_i=epsilon^(1+(i/(N-1))*alpha).
        clip_grad_norm (float): gradient clipping threshold.
        run_step (int): the number of total steps.
        alpha (float): prioritization exponent.
        beta (float): initial value of degree to use importance sampling.
        learn_period (int): period to train (unit: step)
        uniform_sample_prob (float): ratio of uniform random sampling.
        n_step: number of steps in multi-step Q learning.
    """

    def __init__(
        self,
        # ApeX
        epsilon=0.4,
        epsilon_alpha=7.0,
        clip_grad_norm=40.0,
        # PER
        alpha=0.6,
        beta=0.4,
        learn_period=4,
        uniform_sample_prob=1e-3,
        # MultiStep
        n_step=4,
        **kwargs
    ):
        super(ApeX, self).__init__(**kwargs)
        # ApeX
        self.epsilon = epsilon
        self.epsilon_alpha = epsilon_alpha
        self.clip_grad_norm = clip_grad_norm
        self.num_transitions = 0

        # PER
        self.alpha = alpha
        self.beta = beta
        self.learn_period = learn_period
        self.learn_period_stamp = 0
        self.uniform_sample_prob = uniform_sample_prob
        self.beta_add = (1 - beta) / self.run_step

        # MultiStep
        self.n_step = n_step
        self.memory = PERBuffer(self.buffer_size, uniform_sample_prob)
        self.tmp_buffer = deque(maxlen=n_step + 1)

    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training)
        epsilon = self.epsilon if training else self.epsilon_eval

        q = self.network(self.as_tensor(state))
        if np.random.random() < epsilon:
            batch_size = (
                state[0].shape[0] if isinstance(state, list) else state.shape[0]
            )
            action = np.random.randint(0, self.action_size, size=(batch_size, 1))
        else:
            action = torch.argmax(q, -1, keepdim=True).cpu().numpy()
        q = np.take(q.cpu().numpy(), action)
        return {"action": action, "q": q}

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

        eye = torch.eye(self.action_size).to(self.device)
        one_hot_action = eye[action.view(-1).long()]
        q = (self.network(state) * one_hot_action).sum(1, keepdims=True)

        with torch.no_grad():
            max_Q = torch.max(q).item()
            next_q = self.network(next_state)
            max_a = torch.argmax(next_q, axis=1)
            max_one_hot_action = eye[max_a.long()]

            next_target_q = self.target_network(next_state)
            target_q = (next_target_q * max_one_hot_action).sum(1, keepdims=True)

            for i in reversed(range(self.n_step)):
                target_q = reward[:, i] + (1 - done[:, i]) * self.gamma * target_q

        # Update sum tree
        td_error = abs(target_q - q)
        p_j = torch.pow(td_error, self.alpha)
        for i, p in zip(indices, p_j):
            self.memory.update_priority(p.item(), i)

        weights = torch.unsqueeze(torch.FloatTensor(weights).to(self.device), -1)

        loss = (weights * (td_error**2)).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad_norm)
        self.optimizer.step()

        self.num_learn += 1

        result = {
            "loss": loss.item(),
            "max_Q": max_Q,
            "sampled_p": sampled_p,
            "mean_p": mean_p,
            "num_learn": self.num_learn,
            "num_transitions": self.num_transitions,
        }

        return result

    def process(self, transitions, step):
        result = {}
        self.num_transitions += len(transitions)

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

    def set_distributed(self, id):
        assert self.num_workers > 1

        self.epsilon = self.epsilon ** (
            1 + (id / (self.num_workers - 1)) * self.epsilon_alpha
        )
        return self

    def interact_callback(self, transition):
        _transition = {}
        self.tmp_buffer.append(transition)
        if len(self.tmp_buffer) == self.tmp_buffer.maxlen:
            _transition["state"] = self.tmp_buffer[0]["state"]
            _transition["action"] = self.tmp_buffer[0]["action"]
            _transition["next_state"] = self.tmp_buffer[-1]["state"]

            for key in self.tmp_buffer[0].keys():
                if key not in ["state", "action", "next_state"]:
                    _transition[key] = np.stack(
                        [t[key] for t in self.tmp_buffer][:-1], axis=1
                    )

            target_q = self.tmp_buffer[-1]["q"]
            for i in reversed(range(self.n_step)):
                target_q = (
                    self.tmp_buffer[i]["reward"]
                    + (1 - self.tmp_buffer[i]["done"]) * self.gamma * target_q
                )
            priority = abs(target_q - self.tmp_buffer[0]["q"])

            _transition["priority"] = priority
            del _transition["q"]

        return _transition
