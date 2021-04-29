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
    