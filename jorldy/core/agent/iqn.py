import torch

torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
import numpy as np

from core.network import Network
from core.optimizer import Optimizer
from .dqn import DQN


class IQN(DQN):
    """Implicit quantile network (IQN) agent.

    Args:
        state_size (int): dimension of state.
        action_size (int): dimension of action.
        network (str): key of network class in _network_dict.txt.
        head (str): key of head in _head_dict.txt.
        optim_config (dict): dictionary of the optimizer info.
            (key: 'name', value: name of optimizer)
        num_sample (int): the number of sample points
        embedding_dim (int): dimension of sample point embedding.
        sample_min (float): quantile minimum thresholds (tau_min).
        sample_max (float): quantile maximum thresholds (tau_max).
    """

    def __init__(
        self,
        state_size,
        action_size,
        network="iqn",
        head="mlp",
        optim_config={"name": "adam"},
        num_sample=64,
        embedding_dim=64,
        sample_min=0.0,
        sample_max=1.0,
        **kwargs,
    ):
        super(IQN, self).__init__(
            state_size, action_size, network=network, head=head, **kwargs
        )
        self.network = Network(
            network, state_size, action_size, embedding_dim, num_sample, head=head
        ).to(self.device)
        self.target_network = Network(
            network, state_size, action_size, embedding_dim, num_sample, head=head
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = Optimizer(**optim_config, params=self.network.parameters())

        self.action_size = action_size
        self.num_support = num_sample
        self.embedding_dim = embedding_dim
        self.sample_min = sample_min
        self.sample_max = sample_max

    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training)
        epsilon = self.epsilon if training else self.epsilon_eval
        sample_min = 0 if training else self.sample_min
        sample_max = 1 if training else self.sample_max

        if np.random.random() < epsilon:
            batch_size = (
                state[0].shape[0] if isinstance(state, list) else state.shape[0]
            )
            action = np.random.randint(0, self.action_size, size=(batch_size, 1))
        else:
            logits, _ = self.network(self.as_tensor(state), sample_min, sample_max)
            _, q_action = self.logits2Q(logits)
            action = torch.argmax(q_action, -1, keepdim=True).cpu().numpy()
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

        # Get Theta Pred, Tau
        logit, tau = self.network(state)
        logits, q_action = self.logits2Q(logit)
        action_eye = torch.eye(self.action_size, device=self.device)
        action_onehot = action_eye[action.long()]

        theta_pred = action_onehot @ logits
        tau = torch.transpose(tau, 1, 2).contiguous()

        with torch.no_grad():
            # Get Theta Target
            logit_next, _ = self.network(next_state)
            _, q_next = self.logits2Q(logit_next)

            logit_target, _ = self.target_network(next_state)
            logits_target, _ = self.logits2Q(logit_target)

            max_a = torch.argmax(q_next, axis=-1, keepdim=True)
            max_a_onehot = action_eye[max_a.long()]

            theta_target = reward + (1 - done) * self.gamma * torch.squeeze(
                max_a_onehot @ logits_target, 1
            )
            theta_target = torch.unsqueeze(theta_target, 2)

        error_loss = theta_target - theta_pred
        huber_loss = F.smooth_l1_loss(
            *torch.broadcast_tensors(theta_pred, theta_target), reduction="none"
        )

        # Get Loss
        loss = torch.where(error_loss < 0.0, 1 - tau, tau) * huber_loss
        loss = torch.mean(torch.sum(loss, axis=2))

        max_Q = torch.max(q_action).item()
        max_logit = torch.max(logit).item()
        min_logit = torch.min(logit).item()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.num_learn += 1

        result = {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "max_Q": max_Q,
            "max_logit": max_logit,
            "min_logit": min_logit,
        }
        return result

    def logits2Q(self, logits):
        _logits = torch.transpose(logits, 1, 2).contiguous()

        q_action = torch.mean(_logits, dim=-1)
        return _logits, q_action
