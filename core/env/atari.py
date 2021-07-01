import gym
import numpy as np

# https://pypi.org/project/gym-super-mario-bros/
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from .utils import ImgProcessor
from .base import BaseEnv

COMMON_VERSION = 'Deterministic-v4'

class Atari(BaseEnv):
    def __init__(self,
                 name,
                 render=False,
                 gray_img=True,
                 img_width=84,
                 img_height=84,
                 stack_frame=4,
                 id=0,
                 life_key='ale.lives',
                 no_op=False,
                 reward_clip=True,
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
        self.no_op = no_op
        self.no_op_max = 30
        self.reward_clip = reward_clip
        
        print(f"{name} Start!")
        print(f"state size: {self.state_size}")
        print(f"action size: {self.action_size}")
        
    def reset(self):
        self.env.reset()
        state, reward, _, info = self.env.step(1)
        self.score = reward
        self.life = info[self.life_key]
        
        if self.no_op:
            for _ in range(np.random.randint(0, self.no_op_max)):
                state, reward, _, info = self.env.step(0)
                self.score += reward
                if self.life != info[self.life_key]:
                    state, reward, _, _ = self.env.step(1)
                    self.score += reward
                    self.life = info[self.life_key]

        state = self.img_processor.convert_img(state)
        self.stacked_state = np.tile(state, (self.stack_frame,1,1))
        state = np.expand_dims(self.stacked_state, 0)
        return state

    def step(self, action):
        if self.render:
            self.env.render()
            
        next_state, reward, done, info = self.env.step(np.asscalar(action))
        self.score += reward 
        
        if self.life != info[self.life_key] and not done:
            next_state, _reward, _, _ = self.env.step(1)
            self.life = info[self.life_key]
            self.score += _reward 
        next_state = self.img_processor.convert_img(next_state)
        self.stacked_state = np.concatenate((self.stacked_state[self.num_channel:], next_state), axis=0)
        
        next_state, reward, done = map(lambda x: np.expand_dims(x, 0), [self.stacked_state, [reward], [done]])

        if self.reward_clip:
            reward = np.clip(reward, -1., 1.)
        
        return (next_state, reward, done)
    
    def close(self):
        self.env.close()
    
    def recordable(self):
        return True
    
    def get_frame(self):
        return self.env.ale.getScreenRGB2()

class Breakout(Atari):
    def __init__(self, **kwargs):
        super(Breakout, self).__init__(f"Breakout{COMMON_VERSION}", **kwargs)

class Pong(Atari):
    def __init__(self, **kwargs):
        super(Pong, self).__init__(f"Pong{COMMON_VERSION}", **kwargs)

class Asterix(Atari):
    def __init__(self, **kwargs):
        super(Asterix, self).__init__(f"Asterix{COMMON_VERSION}", **kwargs)

class Assault(Atari):
    def __init__(self, **kwargs):
        super(Assault, self).__init__(f"Assualt{COMMON_VERSION}", **kwargs)

class Seaquest(Atari):
    def __init__(self, **kwargs):
        super(Seaquest, self).__init__(f"Seaquest{COMMON_VERSION}", **kwargs)

class Spaceinvaders(Atari):
    def __init__(self, **kwargs):
        super(Spaceinvaders, self).__init__(f"Spaceinvaders{COMMON_VERSION}", **kwargs)

class Alien(Atari):
    def __init__(self, **kwargs):
        super(Alien, self).__init__(f"Alien{COMMON_VERSION}", **kwargs)
        
class CrazyClimber(Atari):
    def __init__(self, **kwargs):
        super(CrazyClimber, self).__init__(f"CrazyClimber{COMMON_VERSION}", **kwargs)

class PrivateEye(Atari):
    def __init__(self, **kwargs):
        super(PrivateEye, self).__init__(f"PrivateEye{COMMON_VERSION}", **kwargs)
        
class MontezumaRevenge(Atari):
    def __init__(self, **kwargs):
        super(MontezumaRevenge, self).__init__(f"MontezumaRevenge{COMMON_VERSION}", **kwargs)
        
class Mario(Atari):
    def __init__(self, **kwargs):
        super(Mario, self).__init__('SuperMarioBros-v2', life_key='life' ,**kwargs)
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.action_size = self.env.action_space.n
        
    def get_frame(self):
        return np.copy(self.env.screen)
    