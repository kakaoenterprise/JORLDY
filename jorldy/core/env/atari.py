import gym
import numpy as np

from .utils import ImgProcessor
from .base import BaseEnv

COMMON_VERSION = "Deterministic-v4"


class _Atari(BaseEnv):
    """Atari environment.

    Args:
        name (str): name of environment in Atari games.
        render (bool): parameter that determine whether to render.
        gray_img (bool): parameter that determine whether to use gray image.
        img_width (int): width of image input.
        img_height (int): height of image input.
        stack_frame (int): the number of stacked frame in one single state.
        life_key (str): key of life query function in emulator.
        no_op (bool): parameter that determine whether or not to operate during the first 30(no_op_max) steps.
        reward_clip (bool): parameter that determine whether to use reward clipping.
        reward_scale (float): reward normalization denominator.
        dead_penatly (bool): parameter that determine whether to use penalty when the agent dies.
    """

    def __init__(
        self,
        name,
        render=False,
        gray_img=True,
        img_width=84,
        img_height=84,
        stack_frame=4,
        life_key="lives",
        no_op=False,
        reward_clip=False,
        reward_scale=None,
        dead_penalty=False,
        **kwargs,
    ):
        self.render = render
        self.gray_img = gray_img
        self.img_width = img_width
        self.img_height = img_height

        self.img_processor = ImgProcessor(gray_img, img_width, img_height)

        self.stack_frame = stack_frame
        self.num_channel = 1 if self.gray_img else 3
        self.stacked_state = np.zeros(
            [self.num_channel * stack_frame, img_height, img_width]
        )

        self.env = gym.make(name)
        self.state_size = [stack_frame, img_height, img_width]
        self.action_size = self.env.action_space.n
        self.score = 0
        self.life = 0
        self.life_key = life_key
        self.no_op = no_op
        self.no_op_max = 30
        self.reward_clip = reward_clip
        self.reward_scale = reward_scale
        self.dead_penalty = dead_penalty

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
                    if self.life > info[self.life_key]:
                        state, reward, _, _ = self.env.step(1)
                        self.score += reward
                    self.life = info[self.life_key]

        state = self.img_processor.convert_img(state)
        self.stacked_state = np.tile(state, (self.stack_frame, 1, 1))
        state = np.expand_dims(self.stacked_state, 0)
        return state

    def step(self, action):
        if self.render:
            self.env.render()

        next_state, reward, done, info = self.env.step(np.asscalar(action))
        self.score += reward

        dead = False
        if self.life != info[self.life_key] and not done:
            if self.life > info[self.life_key]:
                state, _reward, _, _ = self.env.step(1)
                self.score += _reward
                dead = True
            self.life = info[self.life_key]
        next_state = self.img_processor.convert_img(next_state)
        self.stacked_state = np.concatenate(
            (self.stacked_state[self.num_channel :], next_state), axis=0
        )

        if self.reward_clip:
            reward = (
                reward / self.reward_scale if self.reward_scale else np.tanh(reward)
            )

        if dead and self.dead_penalty:
            reward = -1

        next_state, reward, done = map(
            lambda x: np.expand_dims(x, 0), [self.stacked_state, [reward], [done]]
        )
        return (next_state, reward, done)

    def close(self):
        self.env.close()

    def recordable(self):
        return True

    def get_frame(self):
        return self.env.ale.getScreenRGB()


class Breakout(_Atari):
    def __init__(self, **kwargs):
        super(Breakout, self).__init__(f"Breakout{COMMON_VERSION}", **kwargs)


class Pong(_Atari):
    def __init__(self, **kwargs):
        super(Pong, self).__init__(f"Pong{COMMON_VERSION}", **kwargs)


class Asterix(_Atari):
    def __init__(self, **kwargs):
        super(Asterix, self).__init__(f"Asterix{COMMON_VERSION}", **kwargs)


class Assault(_Atari):
    def __init__(self, **kwargs):
        super(Assault, self).__init__(f"Assault{COMMON_VERSION}", **kwargs)


class Seaquest(_Atari):
    def __init__(self, **kwargs):
        super(Seaquest, self).__init__(f"Seaquest{COMMON_VERSION}", **kwargs)


class Spaceinvaders(_Atari):
    def __init__(self, **kwargs):
        super(Spaceinvaders, self).__init__(f"SpaceInvaders{COMMON_VERSION}", **kwargs)


class Alien(_Atari):
    def __init__(self, **kwargs):
        super(Alien, self).__init__(f"Alien{COMMON_VERSION}", **kwargs)


class CrazyClimber(_Atari):
    def __init__(self, **kwargs):
        super(CrazyClimber, self).__init__(f"CrazyClimber{COMMON_VERSION}", **kwargs)


class Enduro(_Atari):
    def __init__(self, **kwargs):
        super(Enduro, self).__init__(f"Enduro{COMMON_VERSION}", **kwargs)


class Qbert(_Atari):
    def __init__(self, **kwargs):
        super(Qbert, self).__init__(f"Qbert{COMMON_VERSION}", **kwargs)


class PrivateEye(_Atari):
    def __init__(self, **kwargs):
        super(PrivateEye, self).__init__(f"PrivateEye{COMMON_VERSION}", **kwargs)


class MontezumaRevenge(_Atari):
    def __init__(self, **kwargs):
        super(MontezumaRevenge, self).__init__(
            f"MontezumaRevenge{COMMON_VERSION}", **kwargs
        )
