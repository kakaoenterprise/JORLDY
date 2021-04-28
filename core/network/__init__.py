from .dqn import *
from .dueling import *
from .noisy import * 
from .iqn import *
from .sac import *
from .reinforce import *
from .ppo import *

class Network:
    dictionary = {
    "dqn": DQN,
    "dqn_cnn": DQN_CNN,
    "dueling": Dueling,
    "dueling_cnn": Dueling_CNN,
    "noisy": Noisy, 
    "noisy_cnn": Noisy_CNN,
    "iqn": IQN,
    "iqn_cnn": IQN_CNN,
    "sac_actor": SACActor,
    "sac_critic": SACCritic,
    "continuous_policy": ContinuousPolicy,
    "discrete_policy": DiscretePolicy,
    "continuous_pi_v": ContinuousPiV,
    "discrete_pi_v": DiscretePiV,
    "continuous_pi_v_cnn": ContinuousPiV_CNN,
    "discrete_pi_v_cnn": DiscretePiV_CNN,
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