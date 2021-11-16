from abc import *
import numpy as np


class BaseBuffer(ABC):
    def __init__(self):
        self.first_store = True

    def check_dim(self, transition):
        print("########################################")
        print("You should check dimension of transition")
        for key, val in transition.items():
            if len(val) > 1:
                for i in range(len(val)):
                    print(f"{key}{i}: {val[i].shape}")
            else:
                print(f"{key}: {val.shape}")
        print("########################################")
        self.first_store = False

    @abstractmethod
    def store(self, transitions):
        """
        Store transitions into buffer.

        Parameter Type
        - transitions: List[Dict]
        """

    @abstractmethod
    def sample(self, batch_size):
        """
        Sample transition data from buffer as much as the batch size.

        Parameter Type
        - batch_size:  int
        - transitions: List[Dict]
        """
        transitions = [{}]
        return transitions

    def stack_transition(self, batch):
        transitions = {}

        for key in batch[0].keys():
            if len(batch[0][key]) > 1:
                # Multimodal
                b_list = []
                for i in range(len(batch[0][key])):
                    tmp_transition = np.stack([b[key][i][0] for b in batch], axis=0)
                    b_list.append(tmp_transition)
                transitions[key] = b_list
            else:
                transitions[key] = np.stack([b[key][0] for b in batch], axis=0)

        return transitions
