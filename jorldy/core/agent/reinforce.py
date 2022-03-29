import torch

torch.backends.cudnn.benchmark = True
from torch.distributions import Normal
import numpy as np
import os

from core.network import Network
from core.optimizer import Optimizer
from core.buffer import RolloutBuffer
from .base import BaseAgent


class REINFORCE(BaseAgent):
    """REINFORCE agent.

    Args:
        state_size (int): dimension of state.
        action_size (int): dimension of action.
        hidden_size (int): dimension of hidden unit.
        network (str): key of network class in _network_dict.txt.
        head (str): key of head in _head_dict.txt.
        optim_config (dict): dictionary of the optimizer info.
        gamma (float): discount factor.
        use_standardization (bool): parameter that determine whether to use standardization for return.
        run_step (int): the number of total steps.
        lr_decay: lr_decay option which apply decayed weight on parameters of network.
        device (str): device to use.
            (e.g. 'cpu' or 'gpu'. None can also be used, and in this case, the cpu is used.)
    """

    def __init__(
        self,
        state_size,
        action_size,
        hidden_size=512,
        network="discrete_policy",
        head="mlp",
        optim_config={"name": "adam"},
        gamma=0.99,
        use_standardization=True,
        run_step=1e6,
        lr_decay=True,
        device=None,
        **kwargs,
    ):
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.action_type = network.split("_")[0]
        assert self.action_type in ["continuous", "discrete"]

        self.network = Network(
            network, state_size, action_size, D_hidden=hidden_size, head=head
        ).to(self.device)
        self.optimizer = Optimizer(**optim_config, params=self.network.parameters())

        self.gamma = gamma
        self.use_standardization = use_standardization
        self.memory = RolloutBuffer()
        self.run_step = run_step
        self.lr_decay = lr_decay

    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training)

        if self.action_type == "continuous":
            mu, std = self.network(self.as_tensor(state))
            z = torch.normal(mu, std) if training else mu
            action = torch.tanh(z)
        else:
            pi = self.network(self.as_tensor(state))
            action = (
                torch.multinomial(pi, 1)
                if training
                else torch.argmax(pi, dim=-1, keepdim=True)
            )
        return {"action": action.cpu().numpy()}

    def learn(self):
        transitions = self.memory.sample()

        state = transitions["state"]
        action = transitions["action"]
        reward = transitions["reward"]

        ret = np.copy(reward)
        for t in reversed(range(len(ret) - 1)):
            ret[t] += self.gamma * ret[t + 1]
        if self.use_standardization:
            ret = (ret - ret.mean()) / (ret.std() + 1e-7)

        state, action, ret = map(lambda x: self.as_tensor(x), [state, action, ret])

        if self.action_type == "continuous":
            mu, std = self.network(state)
            m = Normal(mu, std)
            z = torch.atanh(torch.clamp(action, -1 + 1e-7, 1 - 1e-7))
            log_prob = m.log_prob(z)
        else:
            pi = self.network(state)
            log_prob = torch.log(pi.gather(1, action.long()))
        loss = -(log_prob * ret).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        result = {"loss": loss.item()}
        return result

    def process(self, transitions, step):
        result = {}
        # Process per step
        self.memory.store(transitions)

        # Process per epi
        if transitions[0]["done"]:
            result = self.learn()
            if self.lr_decay:
                self.learning_rate_decay(step)

        return result

    def save(self, path):
        print(f"...Save model to {path}...")
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            os.path.join(path, "ckpt"),
        )

    def load(self, path):
        print(f"...Load model from {path}...")
        checkpoint = torch.load(os.path.join(path, "ckpt"), map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
