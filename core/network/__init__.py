from .dqn import *
from .sac import *

class Network:
    dictionary = {
    "dqn": DQN,
    "dqn_cnn": DQN_CNN,
    "dueling": Dueling,
    "dueling_cnn": Dueling_CNN,
    "sac_actor": SACActor,
    "sac_critic": SACCritic,
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
class TemplateNetwork:
    def __init__(self):
        pass

    def forward(self, x):
        pass
'''