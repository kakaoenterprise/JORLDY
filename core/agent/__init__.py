from .dqn import DQNAgent
from .sac import SACAgent
from .double_dqn import DoubleDQNAgent
from .c51 import C51Agent
from .qrdqn import QRDQNAgent
from .iqn import IQNAgent 

class Agent:
    dictionary = {
    "dqn": DQNAgent,
    "sac": SACAgent,
    "double_dqn": DoubleDQNAgent,
    "c51": C51Agent,
    "qrdqn": QRDQNAgent,
    "iqn": IQNAgent
    }
    
    def __new__(self, name, *args, **kwargs):
        expected_type = str
        if type(name) != expected_type:
            print("### name variable must be string! ###")
            raise Exception
        name = name.lower()
        if not name in self.dictionary.keys():
            print(f"### can use only follows {[opt for opt in self.dictionary.keys()]}")
            raise Exception
        return self.dictionary[name](*args, **kwargs)


'''
class BaseAgent:
    def __init__(self):
        pass

    def act(self, state):
        return action

    def learn(self):
        return result
    
    def process(self, transitions):
        result = None
        # Process per step
        
        # Process per epi
        if done :
            pass
        return result

    def save(self, path):
        pass

    def load(self, path):
        pass
'''