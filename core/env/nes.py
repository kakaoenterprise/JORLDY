# https://pypi.org/project/gym-super-mario-bros/
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from .atari import Atari

class Nes(Atari):
    def __init__(self, name, **kwargs):
        super(Nes, self).__init__(name=name, life_key='life', **kwargs)
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        print(f"action size changed: {self.action_size} -> {self.env.action_space.n}")
        self.action_size = self.env.action_space.n
        
    def get_frame(self):
        return np.copy(self.env.screen)

class Mario(Nes):
    def __init__(self, **kwargs):
        super(Mario, self).__init__('SuperMarioBros-v2', **kwargs)

if __name__=="__main__":
    env = Mario()