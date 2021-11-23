import torch

torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
import numpy as np

from .dqn import DQN


class QRDQN(DQN):
    """Quantile Regression DQN (QR-DQN) agent.

    Args:
        state_size (int): dimension of state.
        action_size (int): dimension of action.
        num_support (int): the number of supports.
    """

    def __init__(self, state_size, action_size, num_support=200, **kwargs):
        super(QRDQN, self).__init__(state_size, action_size * num_support, **kwargs)

        self.action_size = action_size
        self.num_support = num_support

        # Get tau
        min_tau = 1 / (2 * self.num_support)
        max_tau = (2 * self.num_support + 1) / (2 * self.num_support)
        self.tau = torch.arange(
            min_tau, max_tau, 1 / self.num_support, device=self.device
        ).view(1, self.num_support)
        self.inv_tau = 1 - self.tau

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
            logits = self.network(self.as_tensor(state))
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

        # Get Theta Pred
        logit = self.network(state)
        logits, q_action = self.logits2Q(logit)
        action_eye = torch.eye(self.action_size, device=self.device)
        action_onehot = action_eye[action.long()]

        theta_pred = action_onehot @ logits

        with torch.no_grad():
            # Get Theta Target
            logit_next = self.network(next_state)
            _, q_next = self.logits2Q(logit_next)

            logit_target = self.target_network(next_state)
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
        loss = torch.where(error_loss < 0.0, self.inv_tau, self.tau) * huber_loss
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
        _logits = logits.view(logits.shape[0], self.action_size, self.num_support)
        q_action = torch.mean(_logits, dim=-1)
        return _logits, q_action
