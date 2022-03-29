from collections import deque
import torch

torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
import numpy as np

from core.network import Network
from core.optimizer import Optimizer
from core.buffer import PERBuffer
from .rainbow import Rainbow


class RainbowIQN(Rainbow):
    """Rainbow IQN agent.

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
        num_sample (int): number of sample points
        embedding_dim (int): dimension of sample point embedding.
        sample_min (float): quantile minimum thresholds (tau_min).
        sample_max (float): quantile maximum thresholds (tau_max).
        device (str): device to use. (e.g. 'cpu' or 'gpu'. None can also be used, and in this case, the cpu is used.)
    """

    def __init__(
        self,
        state_size,
        action_size,
        hidden_size=512,
        network="rainbow_iqn",
        head="mlp",
        optim_config={"name": "adam"},
        gamma=0.99,
        explore_ratio=0.1,
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
        # IQN
        num_sample=64,
        embedding_dim=64,
        sample_min=0.0,
        sample_max=1.0,
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
            embedding_dim,
            num_sample,
            noise_type,
            D_hidden=hidden_size,
            head=head,
        ).to(self.device)
        self.target_network = Network(
            network,
            state_size,
            action_size,
            embedding_dim,
            num_sample,
            noise_type,
            D_hidden=hidden_size,
            head=head,
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = Optimizer(**optim_config, params=self.network.parameters())
        self.gamma = gamma
        self.explore_step = run_step * explore_ratio
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

        # IQN
        self.num_sample = num_sample
        self.embedding_dim = embedding_dim
        self.sample_min = sample_min
        self.sample_max = sample_max

        # MultiStep
        self.memory = PERBuffer(buffer_size, uniform_sample_prob)

    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training)
        sample_min = 0 if training else self.sample_min
        sample_max = 1 if training else self.sample_max

        if training and self.memory.size < max(self.batch_size, self.start_train_step):
            batch_size = (
                state[0].shape[0] if isinstance(state, list) else state.shape[0]
            )
            action = np.random.randint(0, self.action_size, size=(batch_size, 1))
        else:
            logits, _ = self.network(
                self.as_tensor(state), training, sample_min, sample_max
            )
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

        # Get Theta Pred, Tau
        logit, tau = self.network(state, True)
        logits, q_action = self.logits2Q(logit)
        action_eye = torch.eye(self.action_size, device=self.device)
        action_onehot = action_eye[action.long()]

        theta_pred = action_onehot @ logits
        tau = torch.transpose(tau, 1, 2).contiguous()

        with torch.no_grad():
            # Get Theta Target
            logit_next, _ = self.network(next_state, True)
            _, q_next = self.logits2Q(logit_next)

            logit_target, _ = self.target_network(next_state, True)
            logits_target, _ = self.logits2Q(logit_target)

            max_a = torch.argmax(q_next, axis=-1, keepdim=True)
            max_a_onehot = action_eye[max_a.long()]

            theta_target = torch.squeeze(max_a_onehot @ logits_target, 1)
            for i in reversed(range(self.n_step)):
                theta_target = (
                    reward[:, i] + (1 - done[:, i]) * self.gamma * theta_target
                )
            theta_target = torch.unsqueeze(theta_target, 2)

        error_loss = theta_target - theta_pred
        huber_loss = F.smooth_l1_loss(
            *torch.broadcast_tensors(theta_pred, theta_target), reduction="none"
        )

        # Get Loss
        loss = torch.where(error_loss < 0.0, 1 - tau, tau) * huber_loss
        loss = torch.mean(torch.sum(loss, axis=2), axis=1)

        max_Q = torch.max(q_action).item()
        max_logit = torch.max(logit).item()
        min_logit = torch.min(logit).item()

        # PER
        p_j = torch.pow(loss, self.alpha)

        for i, p in zip(indices, p_j):
            self.memory.update_priority(p.item(), i)

        weights = torch.unsqueeze(torch.FloatTensor(weights).to(self.device), -1)

        loss = (weights * loss).mean()

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

    def logits2Q(self, logits):
        _logits = torch.transpose(logits, 1, 2).contiguous()

        q_action = torch.mean(_logits, dim=-1)
        return _logits, q_action
