import torch

torch.backends.cudnn.benchmark = True

from .dqn import DQN
from core.buffer import PERBuffer


class PER(DQN):
    """Prioritized experience replay (PER) agent.

    Args:
        alpha (float): priority exponent.
        beta (float): initial value of degree to use importance sampling.
        learn_period (int): period to train (unit: step)
        uniform_sample_prob (float): ratio of uniform random sampling.
        run_step (int): the number of total steps.
    """

    def __init__(
        self,
        alpha=0.6,
        beta=0.4,
        learn_period=16,
        uniform_sample_prob=1e-3,
        run_step=1e6,
        **kwargs
    ):
        super(PER, self).__init__(run_step=run_step, **kwargs)
        self.memory = PERBuffer(self.buffer_size, uniform_sample_prob)
        self.alpha = alpha
        self.beta = beta
        self.beta_add = (1 - beta) / run_step
        self.learn_period = learn_period
        self.learn_period_stamp = 0

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
            max_eye = torch.eye(self.action_size).to(self.device)
            max_one_hot_action = eye[max_a.view(-1).long()]

            next_target_q = self.target_network(next_state)
            target_q = reward + (next_target_q * max_one_hot_action).sum(
                1, keepdims=True
            ) * (self.gamma * (1 - done))

        # Update sum tree
        td_error = abs(target_q - q)
        p_j = torch.pow(td_error, self.alpha)
        for i, p in zip(indices, p_j):
            self.memory.update_priority(p.item(), i)

        weights = torch.unsqueeze(torch.FloatTensor(weights).to(self.device), -1)

        loss = (weights * (td_error**2)).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.num_learn += 1

        result = {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "beta": self.beta,
            "max_Q": max_Q,
            "sampled_p": sampled_p,
            "mean_p": mean_p,
        }
        return result

    def process(self, transitions, step):
        result = {}

        # Process per step
        self.memory.store(transitions)
        delta_t = step - self.time_t
        self.time_t = step
        self.target_update_stamp += delta_t
        self.learn_period_stamp += delta_t

        # Annealing beta
        self.beta = min(1.0, self.beta + (self.beta_add * delta_t))

        if (
            self.learn_period_stamp >= self.learn_period
            and self.memory.size >= self.batch_size
            and self.time_t >= self.start_train_step
        ):
            result = self.learn()
            if self.lr_decay:
                self.learning_rate_decay(step)
            self.learn_period_stamp -= self.learn_period

        # Process per step if train start
        if self.num_learn > 0:
            self.epsilon_decay(delta_t)

            if self.target_update_stamp >= self.target_update_period:
                self.update_target()
                self.target_update_stamp -= self.target_update_period

        return result
