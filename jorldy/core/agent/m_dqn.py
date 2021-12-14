import torch

torch.backends.cudnn.benchmark = True
import torch.nn.functional as F

from .dqn import DQN
from .utils import stable_scaled_log_softmax, stable_softmax


class M_DQN(DQN):
    def __init__(self, alpha=0.9, tau=0.03, l_0=-1, **kwargs):
        super(M_DQN, self).__init__(**kwargs)

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

        eye = torch.eye(self.action_size).to(self.device)
        one_hot_action = eye[action.view(-1).long()]
        q = (self.network(state) * one_hot_action).sum(1, keepdims=True)

        with torch.no_grad():
            max_Q = torch.max(q).item()
            next_target_q = self.target_network(next_state)

            ############################################ M-DQN ############################################
            target_q = self.target_network(state)
            log_policy = (
                stable_scaled_log_softmax(target_q, self.tau) * one_hot_action
            ).sum(-1, keepdims=True)
            clipped_log_policy = torch.clip(log_policy, min=self.l_0, max=0)

            next_log_policy = stable_scaled_log_softmax(next_target_q, self.tau)
            next_policy = stable_softmax(next_target_q, self.tau)

            munchausen_term = self.alpha * clipped_log_policy
            maximum_entropy_term = (
                next_policy * (next_target_q - next_log_policy)
            ).sum(-1, keepdims=True)

            target_q = (
                reward
                + munchausen_term
                + (1 - done) * self.gamma * maximum_entropy_term
            )
            ###############################################################################################

        loss = F.smooth_l1_loss(q, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.num_learn += 1

        result = {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "max_Q": max_Q,
        }
        return result
