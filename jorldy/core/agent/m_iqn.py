import torch

torch.backends.cudnn.benchmark = True
import torch.nn.functional as F

from .iqn import IQN
from .utils import stable_scaled_log_softmax, stable_softmax


class M_IQN(IQN):
    def __init__(self, alpha=0.9, tau=0.03, l_0=-1, **kwargs):
        super(M_IQN, self).__init__(**kwargs)

        self.alpha = alpha
        self.tau = tau
        self.l_0 = l_0

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
            logits_target, next_target_q = self.logits2Q(logit_target)

            max_a = torch.argmax(q_next, axis=-1, keepdim=True)
            max_a_onehot = action_eye[max_a.long()]

            ############################################ M-IQN ############################################
            logit, _ = self.network(state)
            _, target_q = self.logits2Q(logit)

            log_policy = (
                stable_scaled_log_softmax(target_q, self.tau) * action_onehot.squeeze()
            ).sum(-1, keepdims=True)
            clipped_log_policy = torch.clip(log_policy, min=self.l_0, max=0)

            munchausen_term = self.alpha * clipped_log_policy

            next_log_policy = (
                stable_scaled_log_softmax(next_target_q, self.tau)
                .unsqueeze(2)
                .repeat(1, 1, self.num_support)
            )
            next_policy = (
                stable_softmax(next_target_q, self.tau)
                .unsqueeze(2)
                .repeat(1, 1, self.num_support)
            )

            maximum_entropy_term = (
                next_policy * (logits_target - next_log_policy)
            ).sum(1)

            theta_target = (
                reward
                + munchausen_term
                + (1 - done) * self.gamma * maximum_entropy_term
            )
            ###############################################################################################

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
