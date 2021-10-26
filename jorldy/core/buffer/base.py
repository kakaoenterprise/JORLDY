from abc import *

class BaseBuffer(ABC):
    def __init__(self):
        self.first_store = True

    def check_dim(self, transition):
        print("########################################")
        print("You should check dimension of transition")
        for key, val in transition.items():
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
        
