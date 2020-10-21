import gym
import numpy as np
from .utils import ImgProcessor

class Breakout:
    def __init__(self, mode='discrete', 
                 gray_img=True,
                 img_width=80,
                 img_height=80,
                 stack_frame=4,
                 ):
        
        self.img_processor = ImgProcessor()
        
        self.gray_img=gray_img
        self.img_width=img_width
        self.img_height=img_height
        self.stack_frame=stack_frame
        self.num_channel = 1 if self.gray_img else 3 
        
        self.stacked_state = np.zeros([self.num_channel*self.stack_frame, self.img_height, self.img_width])
        
        self.env = gym.make('Breakout-v0')
        self.mode = mode
        self.state_size = [img_width, img_height, stack_frame]
        self.action_size = self.env.action_space.n if mode=='discrete' else 1
        self.score = 0

    def reset(self):
        self.score = 0
        state = self.env.reset()
        state = self.img_processor.convert_img(state, self.gray_img, self.img_width, self.img_height)
        
        self.stacked_state = np.tile(state, (self.stack_frame,1,1))
        return self.stacked_state

    def step(self, action):
        if self.mode == 'continuous':
             action = 0 if action < 0 else 1
        next_state, reward, done, info = self.env.step(action)
        
        next_state = self.img_processor.convert_img(next_state, self.gray_img, self.img_width, self.img_height)
        
        self.stacked_state = np.concatenate((self.stacked_state[self.num_channel:], next_state), axis=0)
        
        self.score += reward 
        return (self.stacked_state, reward, done)

    def close(self):
        self.env.close()
    
