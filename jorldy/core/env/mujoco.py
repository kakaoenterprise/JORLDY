import mujoco_py
import gym
import numpy as np
import cv2

from .base import BaseEnv


class _Mujoco(BaseEnv):
    """Mujoco environment.

    Args:
        name (str): name of environment in Mujoco envs.
        render (bool): parameter that determine whether to render.
    """

    def __init__(
        self,
        name,
        render=False,
        **kwargs,
    ):
        self.render = render

        self.env = gym.make(name)

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.action_type = "continuous"
        self.score = 0

        print(f"{name} Start!")
        print(f"state size: {self.state_size}")
        print(f"action size: {self.action_size}")

    def reset(self):
        self.score = 0
        state = self.env.reset()
        state = np.expand_dims(state, 0)  # for (1, state_size)
        return state

    def step(self, action):
        if self.render:
            self.env.render()

        action = ((action + 1.0) * 0.5) * (
            self.env.action_space.high - self.env.action_space.low
        ) + self.env.action_space.low
        action = np.reshape(action, self.env.action_space.shape)
        next_state, reward, done, info = self.env.step(action)
        self.score += reward

        next_state, reward, done = map(
            lambda x: np.expand_dims(x, 0), [next_state, [reward], [done]]
        )
        return (next_state, reward, done)

    def close(self):
        self.env.close()

    def recordable(self):
        try:
            self.get_frame()
            return True
        except Exception as e:
            return False

    def get_frame(self):
        raw_image = self.env.render(mode="rgb_array")
        return cv2.resize(raw_image, dsize=(256, 256))


class HalfCheetah(_Mujoco):
    def __init__(self, **kwargs):
        super(HalfCheetah, self).__init__(f"HalfCheetah-v3", **kwargs)


class Ant(_Mujoco):
    def __init__(self, **kwargs):
        super(Ant, self).__init__(f"Ant-v3", **kwargs)


class Hopper(_Mujoco):
    def __init__(self, **kwargs):
        super(Hopper, self).__init__(f"Hopper-v3", **kwargs)


class Humanoid(_Mujoco):
    def __init__(self, **kwargs):
        super(Humanoid, self).__init__(f"Humanoid-v3", **kwargs)


class HumanoidStandup(_Mujoco):
    def __init__(self, **kwargs):
        super(HumanoidStandup, self).__init__(f"HumanoidStandup-v2", **kwargs)


class InvertedDoublePendulum(_Mujoco):
    def __init__(self, **kwargs):
        super(InvertedDoublePendulum, self).__init__(
            f"InvertedDoublePendulum-v2", **kwargs
        )


class InvertedPendulum(_Mujoco):
    def __init__(self, **kwargs):
        super(InvertedPendulum, self).__init__(f"InvertedPendulum-v2", **kwargs)


class Reacher(_Mujoco):
    def __init__(self, **kwargs):
        super(Reacher, self).__init__(f"Reacher-v2", **kwargs)


class Swimmer(_Mujoco):
    def __init__(self, **kwargs):
        super(Swimmer, self).__init__(f"Swimmer-v3", **kwargs)


class Walker(_Mujoco):
    def __init__(self, **kwargs):
        super(Walker, self).__init__(f"Walker2d-v3", **kwargs)
