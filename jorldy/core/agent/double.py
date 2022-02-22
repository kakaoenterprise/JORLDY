import torch

torch.backends.cudnn.benchmark = True
import torch.nn.functional as F

from .dqn import DQN


class Double(DQN):
    def __init__(self, **kwargs):
        super(Double, self).__init__(**kwargs)

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
            next_q = self.network(next_state)
            max_a = torch.argmax(next_q, axis=1)
            max_one_hot_action = eye[max_a.view(-1).long()]

            next_target_q = self.target_network(next_state)
            target_q = reward + (next_target_q * max_one_hot_action).sum(
                1, keepdims=True
            ) * (self.gamma * (1 - done))

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
