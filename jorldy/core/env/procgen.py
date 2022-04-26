import cv2
import numpy as np
from procgen import ProcgenEnv

from .utils import ImgProcessor
from .base import BaseEnv


class _Procgen(BaseEnv):
    """Procgen environment.

    Args:
        name (str): name of environment in Procgen games.
        render (bool): parameter that determine whether to render.
        gray_img (bool): parameter that determine whether to use gray image.
        img_width (int): width of image input.
        img_height (int): height of image input.
        stack_frame (int): the number of stacked frame in one single state.
        no_op (bool): parameter that determine whether or not to operate during the first 30(no_op_max) steps.
        skip_frame (int) : the number of skipped frame.
        reward_clip (bool): parameter that determine whether to use reward clipping.
    """

    def __init__(
        self,
        name,
        render=False,
        gray_img=True,
        img_width=64,
        img_height=64,
        stack_frame=4,
        no_op=True,
        skip_frame=4,
        reward_clip=True,
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

        self.env = ProcgenEnv(1, name, render_mode="rgb_array")

        self.state_size = [self.num_channel * stack_frame, img_height, img_width]
        self.action_size = self.env.action_space.n
        self.action_type = "discrete"
        self.score = 0
        self.no_op = no_op
        self.no_op_max = 30
        assert isinstance(skip_frame, int) and skip_frame > 0
        self.skip_frame = skip_frame
        self.skip_frame_buffer = np.zeros(
            (2,) + self.env.observation_space["rgb"].shape, dtype=np.uint8
        )
        self.reward_clip = reward_clip

        print(f"{name} Start!")
        print(f"state size: {self.state_size}")
        print(f"action size: {self.action_size}")

    def reset(self):
        self.env.reset()
        total_reward = 0
        obs = self.env.reset()

        if self.no_op:
            num_no_op = np.random.randint(1, self.no_op_max)
            for i in range(num_no_op):
                obs, reward, done, info = self.env.step(np.zeros(1))
                total_reward += reward[0]
                if done:
                    obs = self.env.reset()

        state = obs["rgb"][0]
        self.score = total_reward

        state = self.img_processor.convert_img(state)
        self.stacked_state = np.tile(state, (self.stack_frame, 1, 1))
        state = np.expand_dims(self.stacked_state, 0)
        return state

    def step(self, action):
        if self.render:
            self.env.render()

        total_reward = 0
        for i in range(self.skip_frame):
            next_obs, reward, done, info = self.env.step(action.reshape((1,)))
            next_state = next_obs["rgb"][0]
            total_reward += reward

            if i == self.skip_frame - 2:
                self.skip_frame_buffer[0] = next_state
            if i == self.skip_frame - 1:
                self.skip_frame_buffer[1] = next_state

            if done:
                break

        next_state = self.skip_frame_buffer.max(axis=0)
        next_state = self.img_processor.convert_img(next_state)
        self.stacked_state = np.concatenate(
            (self.stacked_state[self.num_channel :], next_state), axis=0
        )

        if self.reward_clip:
            total_reward = np.sign(total_reward)

        next_state, total_reward, done = map(
            lambda x: np.expand_dims(x, 0), [self.stacked_state, total_reward, done]
        )
        return (next_state, total_reward, done)

    def close(self):
        self.env.close()

    def recordable(self):
        return True

    def get_frame(self):
        raw_image = self.env.render(mode="rgb_array")
        return cv2.resize(raw_image, dsize=(256, 256))


class Coinrun(_Procgen):
    def __init__(self, **kwargs):
        super(Coinrun, self).__init__("coinrun", **kwargs)


class Bigfish(_Procgen):
    def __init__(self, **kwargs):
        super(Bigfish, self).__init__("bigfish", **kwargs)


class Bossfight(_Procgen):
    def __init__(self, **kwargs):
        super(Bossfight, self).__init__("bossfight", **kwargs)


class Caveflyer(_Procgen):
    def __init__(self, **kwargs):
        super(Caveflyer, self).__init__("caveflyer", **kwargs)


class Chaser(_Procgen):
    def __init__(self, **kwargs):
        super(Chaser, self).__init__("chaser", **kwargs)


class Climber(_Procgen):
    def __init__(self, **kwargs):
        super(Climber, self).__init__("climber", **kwargs)


class Dodgeball(_Procgen):
    def __init__(self, **kwargs):
        super(Dodgeball, self).__init__("dodgeball", **kwargs)


class Fruitbot(_Procgen):
    def __init__(self, **kwargs):
        super(Fruitbot, self).__init__("fruitbot", **kwargs)


class Heist(_Procgen):
    def __init__(self, **kwargs):
        super(Heist, self).__init__("heist", **kwargs)


class Jumper(_Procgen):
    def __init__(self, **kwargs):
        super(Jumper, self).__init__("jumper", **kwargs)


class Leaper(_Procgen):
    def __init__(self, **kwargs):
        super(Leaper, self).__init__("leaper", **kwargs)


class Maze(_Procgen):
    def __init__(self, **kwargs):
        super(Maze, self).__init__("maze", **kwargs)


class Miner(_Procgen):
    def __init__(self, **kwargs):
        super(Miner, self).__init__("miner", **kwargs)


class Ninja(_Procgen):
    def __init__(self, **kwargs):
        super(Ninja, self).__init__("ninja", **kwargs)


class Plunder(_Procgen):
    def __init__(self, **kwargs):
        super(Plunder, self).__init__("plunder", **kwargs)


class Starpilot(_Procgen):
    def __init__(self, **kwargs):
        super(Starpilot, self).__init__("starpilot", **kwargs)
