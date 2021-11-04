import numpy as np
# https://pypi.org/project/gym-super-mario-bros/
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
import numpy as np

from .atari import _Atari

class _Nes(_Atari):
    """Nes environment. 
    
    Args: 
        name (str): name of environment in Nes.
    """
    def __init__(self, name, **kwargs):
        super(_Nes, self).__init__(name=name, life_key='life', **kwargs)
        self.env = JoypadSpace(self.env, RIGHT_ONLY)
        print(f"action size changed: {self.action_size} -> {self.env.action_space.n}")
        self.action_size = self.env.action_space.n
        
    def get_frame(self):
        return np.copy(self.env.screen)

class SuperMarioBros(_Nes):
    def __init__(self, **kwargs):
        reward_scale = 15.
        super(SuperMarioBros, self).__init__('SuperMarioBros-v0', reward_scale=reward_scale, **kwargs)
