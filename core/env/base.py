from abc import *

class BaseEnv(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass
        
    @abstractmethod
    def close(self):
        pass
