from collections import deque
from itertools import islice
import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
import numpy as np

from .base import BaseAgent

class MuZero(BaseAgent):
    """MuZero agent.

    Args:
        -
    """

    def __init__(
        self,
        # MuZero
        -
        **kwargs
    ):
        super(MuZero, self).__init__(network=network, **kwargs)
        

    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training)
        pass

    def learn(self):
        pass
        self.num_learn += 1

        result = {
            "loss": loss.item(),
        }

        return result

    def process(self, transitions, step):
        pass
    
        return result


    def interact_callback(self, transition):
        pass

        return _transition


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
        self.target_network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        

class MCTS():
    def __init__(self):
        pass
    
    def selection(self):
        pass
    
    def expansion(self):
        pass
    
    def backup(self):
        pass
    
    