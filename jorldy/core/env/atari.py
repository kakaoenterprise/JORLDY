import gym
import numpy as np

from .utils import ImgProcessor
from .base import BaseEnv

COMMON_VERSION = "NoFrameskip-v4"


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
        skip_frame (int) : the number of skipped frame.
        reward_clip (bool): parameter that determine whether to use reward clipping.
        episodic_life (bool): parameter that determine done is True when dead is True.
        fire_reset (bool): parameter that determine take action on reset for environments that are fixed until firing.
        train_mode (bool): parameter that determine whether train mode or not.

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
        no_op=True,
        skip_frame=4,
        reward_clip=True,
        episodic_life=True,
        fire_reset=True,
        train_mode=True,
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
        self.state_size = [self.num_channel * stack_frame, img_height, img_width]
        self.action_size = self.env.action_space.n
        self.action_type = "discrete"
        self.score = 0
        self.life = 0
        self.life_key = life_key
        self.no_op = no_op
        self.no_op_max = 30
        assert isinstance(skip_frame, int) and skip_frame > 0
        self.skip_frame = skip_frame
        self.skip_frame_buffer = np.zeros(
            (2,) + self.env.observation_space.shape, dtype=np.uint8
        )
        self.reward_clip = reward_clip
        self.episodic_life = episodic_life
        self.was_real_done = True
        self.fire_reset = fire_reset and (
            self.env.unwrapped.get_action_meanings()[1] == "FIRE"
        )
        self.train_mode = train_mode

        print(f"{name} Start!")
        print(f"state size: {self.state_size}")
        print(f"action size: {self.action_size}")

    def reset(self):
        total_reward = 0
        if self.was_real_done:
            state = self.env.reset()
            self.was_real_done = False
            if self.no_op:
                num_no_op = np.random.randint(1, self.no_op_max)
                for i in range(num_no_op):
                    state, reward, done, info = self.env.step(0)
                    total_reward += reward
                    if done:
                        self.env.reset()
            if self.fire_reset:
                state, reward, done, info = self.env.step(1)
                self.life = info[self.life_key]
                total_reward += reward
        else:
            if self.fire_reset:
                state, reward, _, info = self.env.step(1)
            else:
                state, reward, _, info = self.env.step(0)
            self.life = info[self.life_key]
            total_reward += reward
        self.score = total_reward

        state = self.img_processor.convert_img(state)
        self.stacked_state = np.tile(state, (self.stack_frame, 1, 1))
        state = np.expand_dims(self.stacked_state, 0)
        return state

    def step(self, action):
        if self.render:
            self.env.render()

        dead, total_reward = False, 0
        for i in range(self.skip_frame):
            next_state, reward, done, info = self.env.step(action.item())
            total_reward += reward
            _dead = False
            if self.life != info[self.life_key] and not done:
                if self.life > info[self.life_key]:
                    if self.fire_reset:
                        next_state, reward, _, _ = self.env.step(1)
                        total_reward += reward
                    _dead = True
                self.life = info[self.life_key]

            dead = dead or _dead
            if i == self.skip_frame - 2:
                self.skip_frame_buffer[0] = next_state
            if i == self.skip_frame - 1:
                self.skip_frame_buffer[1] = next_state

            if done:
                self.was_real_done = True
                break

        self.score += total_reward

        next_state = self.skip_frame_buffer.max(axis=0)
        next_state = self.img_processor.convert_img(next_state)
        self.stacked_state = np.concatenate(
            (self.stacked_state[self.num_channel :], next_state), axis=0
        )

        if self.reward_clip:
            total_reward = np.sign(total_reward)

        if self.episodic_life and self.train_mode:
            done = dead or done

        next_state, total_reward, done = map(
            lambda x: np.expand_dims(x, 0), [self.stacked_state, [total_reward], [done]]
        )
        return (next_state, total_reward, done)

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


class BattleZone(_Atari):
    def __init__(self, **kwargs):
        super(BattleZone, self).__init__(f"BattleZone{COMMON_VERSION}", **kwargs)


class Robotank(_Atari):
    def __init__(self, **kwargs):
        super(Robotank, self).__init__(f"Robotank{COMMON_VERSION}", **kwargs)


class MsPacman(_Atari):
    def __init__(self, **kwargs):
        super(MsPacman, self).__init__(f"MsPacman{COMMON_VERSION}", **kwargs)


class TimePilot(_Atari):
    def __init__(self, **kwargs):
        super(TimePilot, self).__init__(f"TimePilot{COMMON_VERSION}", **kwargs)
