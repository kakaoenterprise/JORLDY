import numpy as np
# https://pypi.org/project/gym-super-mario-bros/
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

from .atari import Atari

class Nes(Atari):
    def __init__(self, name, **kwargs):
        super(Nes, self).__init__(name=name, life_key='life', **kwargs)
        self.env = JoypadSpace(self.env, RIGHT_ONLY)
        print(f"action size changed: {self.action_size} -> {self.env.action_space.n}")
        self.action_size = self.env.action_space.n
        
    def get_frame(self):
        return np.copy(self.env.screen)

class Mario(Nes):
    def __init__(self, **kwargs):
        reward_scale = 15.
        super(Mario, self).__init__('SuperMarioBros-v2', reward_scale=reward_scale, **kwargs)

if __name__=="__main__":
    env = Mario()