import torch

torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
import numpy as np

from .dqn import DQN


class C51(DQN):
    """C51 agent.

    Args:
        state_size (int): dimension of state.
        action_size (int): dimension of action.
        v_min (float): minimum value of support.
        v_max (float): maximum value of support.
        num_support (int): number of support.
    """

    def __init__(
        self, state_size, action_size, v_min=-10, v_max=10, num_support=51, **kwargs
    ):
        super(C51, self).__init__(state_size, action_size * num_support, **kwargs)

        self.action_size = action_size
        self.v_min = v_min
        self.v_max = v_max
        self.num_support = num_support
        self.delta_z = (self.v_max - self.v_min) / (self.num_support - 1)
        self.z = torch.linspace(
            self.v_min, self.v_max, self.num_support, device=self.device
        ).view(1, -1)

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

        logit = self.network(state)
        p_logit, q_action = self.logits2Q(logit)

        action_eye = torch.eye(self.action_size, device=self.device)
        action_onehot = action_eye[action.long()]

        p_action = torch.squeeze(action_onehot @ p_logit, 1)

        target_dist = torch.zeros(
            self.batch_size, self.num_support, device=self.device, requires_grad=False
        )
        with torch.no_grad():
            target_p_logit, target_q_action = self.logits2Q(
                self.target_network(next_state)
            )

            target_action = torch.argmax(target_q_action, -1, keepdim=True)
            target_action_onehot = action_eye[target_action.long()]
            target_p_action = torch.squeeze(target_action_onehot @ target_p_logit, 1)

            Tz = reward.expand(-1, self.num_support) + (1 - done) * self.gamma * self.z
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
            target_dist += done * torch.mean(
                l_support_onehot * u_support_onehot + lluu, 1
            )
            target_dist += (1 - done) * torch.sum(target_p_action_binary * lluu, 1)
            target_dist /= torch.clamp(
                torch.sum(target_dist, 1, keepdim=True), min=1e-8
            )

        max_Q = torch.max(q_action).item()
        max_logit = torch.max(logit).item()
        min_logit = torch.min(logit).item()

        loss = -(target_dist * torch.clamp(p_action, min=1e-8).log()).sum(-1).mean()
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
        _logits_max = torch.max(_logits, -1, keepdim=True).values
        p_logit = torch.exp(F.log_softmax(_logits - _logits_max, dim=-1))

        z_action = self.z.expand(p_logit.shape[0], self.action_size, self.num_support)
        q_action = torch.sum(z_action * p_logit, dim=-1)

        return p_logit, q_action
