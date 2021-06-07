import gym
import numpy as np
from .utils import ImgProcessor
from .base import BaseEnv

class Atari(BaseEnv):
    def __init__(self,
                 name,
                 render=False,
                 gray_img=True,
                 img_width=84,
                 img_height=84,
                 stack_frame=4,
                 id=0,
                 life_key='ale.lives'
                 ):
        self.id = id
        self.render=render
        self.gray_img=gray_img
        self.img_width=img_width
        self.img_height=img_height
        self.img_processor = ImgProcessor(gray_img, img_width, img_height)
        self.stack_frame=stack_frame
        self.num_channel = 1 if self.gray_img else 3 
        self.stacked_state = np.zeros([self.num_channel*stack_frame, img_height, img_width])
        
        self.env = gym.make(name)
        self.state_size = [stack_frame, img_height, img_width]
        self.action_size = self.env.action_space.n
        self.score = 0
        self.life = 0
        self.life_key = life_key

    def reset(self):
        self.env.reset()
        state, _, _, info = self.env.step(1)
        state = self.img_processor.convert_img(state)
        self.stacked_state = np.tile(state, (self.stack_frame,1,1))
        self.life = info[self.life_key]
        self.score = 0
        state = np.expand_dims(self.stacked_state, 0)
        return state

    def step(self, action):
        if self.render:
            self.env.render()
        action = np.asscalar(action)
        next_state, reward, done, info = self.env.step(action)
        
        if self.life != info[self.life_key] and not done:
            next_state, _, _, _ = self.env.step(1)
            next_state = self.img_processor.convert_img(next_state)
            self.stacked_state = np.tile(next_state, (self.stack_frame,1,1))
            self.life = info[self.life_key]
        else:
            next_state = self.img_processor.convert_img(next_state)
            self.stacked_state = np.concatenate((self.stacked_state[self.num_channel:], next_state), axis=0)
        
        self.score += reward 
        
        next_state, reward, done = map(lambda x: np.expand_dims(x, 0), [self.stacked_state, [reward], [done]])
        return (next_state, reward, done)

    def close(self):
        self.env.close()

class Breakout(Atari):
    def __init__(self, **kwargs):
        super(Breakout, self).__init__('BreakoutDeterministic-v4', **kwargs)

class Pong(Atari):
    def __init__(self, **kwargs):
        super(Pong, self).__init__('Pong-v0', **kwargs)

class Asterix(Atari):
    def __init__(self, **kwargs):
        super(Asterix, self).__init__('Asterix-v0', **kwargs)

class Assault(Atari):
    def __init__(self, **kwargs):
        super(Assault, self).__init__('AssaultDeterministic-v4', **kwargs)

class Seaquest(Atari):
    def __init__(self, **kwargs):
        super(Seaquest, self).__init__('Seaquest-v0', **kwargs)

class Spaceinvaders(Atari):
    def __init__(self, **kwargs):
        super(Spaceinvaders, self).__init__('SpaceInvaders-v0', **kwargs)

class Alien(Atari):
    def __init__(self, **kwargs):
        super(Alien, self).__init__('Alien-v0', **kwargs)
        
class CrazyClimber(Atari):
    def __init__(self, **kwargs):
        super(CrazyClimber, self).__init__('CrazyClimber-v0', **kwargs)
        
class PrivateEye(Atari):
    def __init__(self, **kwargs):
        super(PrivateEye, self).__init__('PrivateEye-v0', **kwargs)
        
class MontezumaRevenge(Atari):
    def __init__(self, **kwargs):
        super(MontezumaRevenge, self).__init__('MontezumaRevenge-v0', **kwargs)