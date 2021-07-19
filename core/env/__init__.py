import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # for import mlagents
sys.path.append(os.path.abspath('../../'))

from .gym_env import *
from .atari import *
from .ml_agent import * 
from .nes import *

class Env:
    dictionary = {
    #gym_env
    "cartpole": CartPole,
    "pendulum": Pendulum,
    "mountaincar": MountainCar,
    #atari
    "breakout": Breakout,
    "pong": Pong,
    "alien": Alien,
    "asterix": Asterix,
    "assault": Assault,
    "crazyclimber": CrazyClimber,
    "enduro": Enduro,
    "qbert": Qbert,
    "privateeye": PrivateEye,
    "montezuma": MontezumaRevenge,
    "spaceinvaders": Spaceinvaders,
    "seaquest": Seaquest,
    #nes
    "mario": Mario,
    #ml_agent
    "hopper_mlagent": HopperMLAgent,
    "pong_mlagent": PongMLAgent,
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
class BaseEnv:
    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass

    def close(self):
        pass
'''