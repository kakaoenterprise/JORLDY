# https://pypi.org/project/gym-super-mario-bros/

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import numpy as np
from .utils import ImgProcessor
from .base import BaseEnv

class Mario_Env(BaseEnv):
    def __init__(self,
                 name,
                 render=False,
                 gray_img=True,
                 img_width=84,
                 img_height=84,
                 stack_frame=4,
                 id=0,
                 ):
        self.id = id
        self.img_processor = ImgProcessor()
        self.render=render
        self.gray_img=gray_img
        self.img_width=img_width
        self.img_height=img_height
        self.stack_frame=stack_frame
        self.num_channel = 1 if self.gray_img else 3 
        self.stacked_state = np.zeros([self.num_channel*stack_frame, img_height, img_width])
        
        self.env = gym_super_mario_bros.make('SuperMarioBros-v0')
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        
        self.state_size = [stack_frame, img_height, img_width]
        self.action_size = self.env.action_space.n
        self.score = 0
        self.life = 0

    def reset(self):
        self.env.reset()
        state, _, _, info = self.env.step(1)
        state = self.img_processor.convert_img(state, self.gray_img, self.img_width, self.img_height)
        self.stacked_state = np.tile(state, (self.stack_frame,1,1))
        self.life = info['life']
        self.score = 0
        state = np.expand_dims(self.stacked_state, 0)
        return state

    def step(self, action):
        if self.render:
            self.env.render()
        action = np.asscalar(action)
        
        next_state, reward, done, info = self.env.step(action)
        
        print(action)
        print(info)
        print('=============================')
        
        if self.life != info['life'] and not done:
            next_state, _, _, _ = self.env.step(1)
            next_state = self.img_processor.convert_img(next_state, self.gray_img, self.img_width, self.img_height)
            self.stacked_state = np.tile(next_state, (self.stack_frame,1,1))
            self.life = info['life']
        else:
            next_state = self.img_processor.convert_img(next_state, self.gray_img, self.img_width, self.img_height)
            self.stacked_state = np.concatenate((self.stacked_state[self.num_channel:], next_state), axis=0)

        self.score += reward 
        
        next_state, reward, done = map(lambda x: np.expand_dims(x, 0), [self.stacked_state, [reward], [done]])
        return (next_state, reward, done)

    def close(self):
        self.env.close()

class Mario(Mario_Env):
    def __init__(self,
                **kwargs
                ):
        super(Mario, self).__init__('SuperMarioBros-v0', **kwargs)