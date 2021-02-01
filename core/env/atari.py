import gym
import numpy as np
from .utils import ImgProcessor

class Atari:
    def __init__(self,
                 name,
                 render=False,
                 gray_img=True,
                 img_width=84,
                 img_height=84,
                 stack_frame=4,
                 ):
        self.img_processor = ImgProcessor()
        self.render=render
        self.gray_img=gray_img
        self.img_width=img_width
        self.img_height=img_height
        self.stack_frame=stack_frame
        self.num_channel = 1 if self.gray_img else 3 
        self.stacked_state = np.zeros([self.num_channel*stack_frame, img_height, img_width])
        
        self.env = gym.make(name)
        self.state_size = [stack_frame, img_height, img_width]
        self.action_size = self.env.action_space.n
        self.score = 0
        self.life = 0

    def reset(self):
        self.env.reset()
        state, _, _, info = self.env.step(1)
        state = self.img_processor.convert_img(state, self.gray_img, self.img_width, self.img_height)
        self.stacked_state = np.tile(state, (self.stack_frame,1,1))
        self.life = info['ale.lives']
        self.score = 0
        state = np.expand_dims(self.stacked_state, 0)
        return state

    def step(self, action):
        if self.render:
            self.env.render()
        action = np.asscalar(action)
        next_state, reward, done, info = self.env.step(action)
        
        if self.life != info['ale.lives']:
            next_state, _, _, _ = self.env.step(1)
            next_state = self.img_processor.convert_img(next_state, self.gray_img, self.img_width, self.img_height)
            self.stacked_state = np.tile(next_state, (self.stack_frame,1,1))
            self.life = info['ale.lives']
        else:
            next_state = self.img_processor.convert_img(next_state, self.gray_img, self.img_width, self.img_height)
            self.stacked_state = np.concatenate((self.stacked_state[self.num_channel:], next_state), axis=0)
        
        self.score += reward 
        
        next_state, reward, done = map(lambda x: np.expand_dims(x, 0), [self.stacked_state, [reward], [done]])
        return (next_state, reward, done)

    def close(self):
        self.env.close()

class Breakout(Atari):
    def __init__(self, **kwargs):
        super(Breakout, self).__init__('BreakoutDeterministic-v4')

class Pong(Atari):
    def __init__(self, **kwargs):
        super(Pong, self).__init__('Pong-v0')

class Asterix(Atari):
    def __init__(self, **kwargs):
        super(Asterix, self).__init__('Asterix-v0')

        
# class Breakout:
#     def __init__(self, 
#                  render=False,
#                  gray_img=True,
#                  img_width=84,
#                  img_height=84,
#                  stack_frame=4,
#                  ):
#         self.img_processor = ImgProcessor()
#         self.render=render
#         self.gray_img=gray_img
#         self.img_width=img_width
#         self.img_height=img_height
#         self.stack_frame=stack_frame
#         self.num_channel = 1 if self.gray_img else 3 
#         self.stacked_state = np.zeros([self.num_channel*stack_frame, img_height, img_width])
        
#         self.env = gym.make('BreakoutDeterministic-v4')
#         self.state_size = [stack_frame, img_height, img_width]
# #         self.action_size = self.env.action_space.n - 1  # except no-action
#         self.action_size = self.env.action_space.n
#         self.score = 0
#         self.life = 0

#     def reset(self):
#         self.env.reset()
#         state, _, _, info = self.env.step(1)
#         state = self.img_processor.convert_img(state, self.gray_img, self.img_width, self.img_height)
#         self.stacked_state = np.tile(state, (self.stack_frame,1,1))
#         self.life = info['ale.lives']
#         self.score = 0
#         state = np.expand_dims(self.stacked_state, 0)
#         return state

#     def step(self, action):
#         if self.render:
#             self.env.render()
#         action = np.asscalar(action)
# #         next_state, reward, done, info = self.env.step(action+1)
#         next_state, reward, done, info = self.env.step(action)
        
#         if self.life != info['ale.lives']:
#             next_state, _, _, _ = self.env.step(1)
#             next_state = self.img_processor.convert_img(next_state, self.gray_img, self.img_width, self.img_height)
#             self.stacked_state = np.tile(next_state, (self.stack_frame,1,1))
#             self.life = info['ale.lives']
#         else:
#             next_state = self.img_processor.convert_img(next_state, self.gray_img, self.img_width, self.img_height)
#             self.stacked_state = np.concatenate((self.stacked_state[self.num_channel:], next_state), axis=0)
        
#         self.score += reward 
        
#         next_state, reward, done = map(lambda x: np.expand_dims(x, 0), [self.stacked_state, [reward], [done]])
#         return (next_state, reward, done)

#     def close(self):
#         self.env.close()

# class Pong:
#     def __init__(self,
#                  render=False,
#                  gray_img=True,
#                  img_width=84,
#                  img_height=84,
#                  stack_frame=4,
#                  ):
#         self.img_processor = ImgProcessor()
#         self.render=render
#         self.gray_img=gray_img
#         self.img_width=img_width
#         self.img_height=img_height
#         self.stack_frame=stack_frame
#         self.num_channel = 1 if self.gray_img else 3 
#         self.stacked_state = np.zeros([self.num_channel*self.stack_frame, self.img_height, self.img_width])
        
#         self.env = gym.make('Pong-v0')
#         self.state_size = [stack_frame, img_height, img_width]
#         self.action_size = self.env.action_space.n
#         self.score = 0

#     def reset(self):
#         self.env.reset()
#         state, _, _, info = self.env.step(1)
#         state = self.img_processor.convert_img(state, self.gray_img, self.img_width, self.img_height)
#         self.stacked_state = np.tile(state, (self.stack_frame,1,1))
#         self.score = 0
#         state = np.expand_dims(self.stacked_state, 0)
#         return state

#     def step(self, action):
#         if self.render:
#             self.env.render()
#         action = np.asscalar(action)
#         state, reward, done, info = self.env.step(action)

#         state = self.img_processor.convert_img(state, self.gray_img, self.img_width, self.img_height)
#         self.stacked_state = np.concatenate((self.stacked_state[self.num_channel:], state), axis=0)
        
#         self.score += reward 
#         next_state, reward, done = map(lambda x: np.expand_dims(x, 0), [self.stacked_state, [reward], [done]])
#         return (next_state, reward, done)

#     def close(self):
#         self.env.close()
        
# class Asterix:
#     def __init__(self,
#                  render=False,
#                  gray_img=True,
#                  img_width=84,
#                  img_height=84,
#                  stack_frame=4,
#                  ):
#         self.img_processor = ImgProcessor()
#         self.render=render
#         self.gray_img=gray_img
#         self.img_width=img_width
#         self.img_height=img_height
#         self.stack_frame=stack_frame
#         self.num_channel = 1 if self.gray_img else 3 
#         self.stacked_state = np.zeros([self.num_channel*self.stack_frame, self.img_height, self.img_width])
        
#         self.env = gym.make('Asterix-v0')
#         self.state_size = [stack_frame, img_height, img_width]
#         self.action_size = self.env.action_space.n
#         self.score = 0
#         self.life = 0
#         print(self.env.action_space.n)
# #         exit()
        
#     def reset(self):
#         self.env.reset()
#         state, _, _, info = self.env.step(1)
#         state = self.img_processor.convert_img(state, self.gray_img, self.img_width, self.img_height)
#         self.stacked_state = np.tile(state, (self.stack_frame,1,1))
#         self.life = info['ale.lives']
#         print(self.life)
#         exit()
#         self.score = 0
#         state = np.expand_dims(self.stacked_state, 0)
#         return state

#     def step(self, action):
#         if self.render:
#             self.env.render()
#         action = np.asscalar(action)
#         state, reward, done, info = self.env.step(action)

#         state = self.img_processor.convert_img(state, self.gray_img, self.img_width, self.img_height)
#         self.stacked_state = np.concatenate((self.stacked_state[self.num_channel:], state), axis=0)
        
#         self.score += reward 
#         next_state, reward, done = map(lambda x: np.expand_dims(x, 0), [self.stacked_state, [reward], [done]])
#         return (next_state, reward, done)

#     def close(self):
#         self.env.close()
    
