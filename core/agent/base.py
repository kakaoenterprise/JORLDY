from abc import *

class BaseAgent(ABC):
    @abstractmethod
    def act(self, state):
        pass
    
    @abstractmethod
    def learn(self):
        pass
    
    @abstractmethod
    def process(self):
        pass
    
    @abstractmethod
    def save(self, path):
        pass
    
    @abstractmethod
    def load(self, path):
        pass
    
    def sync_in(self, weights):
        self.network.load_state_dict(weights)
    
    def sync_out(self, device="cpu"):
        weights = self.network.state_dict()
        for k, v in weights.items():
            weights[k] = v.to(device) 
        sync_item ={
            "weights": weights,
        }
        return sync_item
    
    def set_distributed(self, id):
        return self