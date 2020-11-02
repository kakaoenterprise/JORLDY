import gym
import numpy as np
from .utils import ImgProcessor

class Breakout:
    def __init__(self, 
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
        
        self.stacked_state = np.zeros([self.num_channel*stack_frame, img_height, img_width])
        
        self.env = gym.make('Breakout-v4')
        self.state_size = [stack_frame, img_height, img_width]
        self.action_size = 3
        self.score = 0
        self.life = 0

    def reset(self):
        self.env.reset()
        state, _, _, info = self.env.step(1)
        state = self.img_processor.convert_img(state, self.gray_img, self.img_width, self.img_height)
        self.stacked_state = np.tile(state, (self.stack_frame,1,1))
        self.life = info['ale.lives']
        self.score = 0
        return self.stacked_state

    def step(self, action):
        self.env.render()
        state, reward, done, info = self.env.step(action+1)
        
        if self.life != info['ale.lives']:
            state, _, _, _ = self.env.step(1)
            state = self.img_processor.convert_img(state, self.gray_img, self.img_width, self.img_height)
            self.stacked_state = np.tile(state, (self.stack_frame,1,1))
            self.life = info['ale.lives']
        else:
            state = self.img_processor.convert_img(state, self.gray_img, self.img_width, self.img_height)
            self.stacked_state = np.concatenate((self.stacked_state[self.num_channel:], state), axis=0)
        
        self.score += reward 
        return (self.stacked_state, reward, done)

    def close(self):
        self.env.close()

class Pong:
    def __init__(self, 
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
        
        self.env = gym.make('Pong-v0')
        self.state_size = [img_width, img_height, stack_frame]
        self.action_size = self.env.action_space.n
        self.score = 0
#         self.life = 0

    def reset(self):
        self.env.reset()
        state, _, _, info = self.env.step(1)
        state = self.img_processor.convert_img(state, self.gray_img, self.img_width, self.img_height)
        self.stacked_state = np.tile(state, (self.stack_frame,1,1))
#         self.life = info['ale.lives']
        self.score = 0
        return self.stacked_state

    def step(self, action):
        state, reward, done, info = self.env.step(action)
#         if self.life != info['ale.lives']:
#             state, _, _, _ = self.env.step(1)
#             state = self.img_processor.convert_img(state, self.gray_img, self.img_width, self.img_height)
#             self.stacked_state = np.tile(state, (self.stack_frame,1,1))
#             self.life = info['ale.lives']
#         else:
        state = self.img_processor.convert_img(state, self.gray_img, self.img_width, self.img_height)
        self.stacked_state = np.concatenate((self.stacked_state[self.num_channel:], state), axis=0)
        
        self.score += reward 
        return (self.stacked_state, reward, done)

    def close(self):
        self.env.close()
    
