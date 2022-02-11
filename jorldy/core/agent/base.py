from abc import *
import torch
import numpy as np


class BaseAgent(ABC):
    @abstractmethod
    def act(self, state):
        """
        Compute action through the state, and return the items to store in the buffer, including the action, in the form of a dictionary.

        Parameter Type / Shape
        - state:        ndarray / (N_batch, *D_state) ex) (1, 4), (1, 4, 84, 84)
        - action:       ndarray / (N_batch, *D_action) ex) (1, 3), (1, 1)
        - action_dict:  dict /
        """
        action = None
        action_dict = {
            "action": action,
        }
        return action_dict

    @abstractmethod
    def learn(self):
        """
        Optimize model, and return the values ​​you want to record from optimization process in the form of a dictionary.

        Parameter Type / Shape
        - result: dict /
        """
        result = {
            "loss": None,
        }
        return result

    @abstractmethod
    def process(self, transitions, step):
        """
        Execute specific tasks at each period, including learn process, and return the result from the learn process.

        Parameter Type / Shape
        result: dict /
        """
        result = {}
        return result

    @abstractmethod
    def save(self, path):
        """
        Save model to path.
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Load model from path.
        """
        pass

    def as_tensor(self, x):
        if isinstance(x, list):
            x = list(
                map(
                    lambda x: torch.as_tensor(
                        x, dtype=torch.float32, device=self.device
                    ),
                    x,
                )
            )
        else:
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        return x

    def sync_in(self, weights):
        self.network.load_state_dict(weights)

    def sync_out(self, device="cpu"):
        weights = self.network.state_dict()
        for k, v in weights.items():
            weights[k] = v.to(device)
        sync_item = {
            "weights": weights,
        }
        return sync_item

    def set_distributed(self, *args, **kwargs):
        return self

    def interact_callback(self, transition):
        return transition

    def learning_rate_decay(self, step, optimizers=None, mode="cosine"):
        if mode == "linear":
            weight = 1 - (step / self.run_step)
        elif mode == "cosine":
            weight = np.cos((np.pi / 2) * (step / self.run_step))
        elif mode == "sqrt":
            weight = (1 - (step / self.run_step)) ** (1 / 2)
        else:
            raise Exception(f"check learning rate decay mode again! => {mode}")

        if optimizers is None:
            optimizers = [self.optimizer]

        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        for optimizer in optimizers:
            for g in optimizer.param_groups:
                g["lr"] = optimizer.defaults["lr"] * weight
