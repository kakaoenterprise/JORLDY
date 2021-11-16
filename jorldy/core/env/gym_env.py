import gym
import numpy as np
from .base import BaseEnv


class _Gym(BaseEnv):
    """Gym environment.

    Args:
        name (str): name of environment in Gym.
        mode (str): type of state and action space. One of ['discrete', 'continuous'].
        render (bool): parameter that determine whether to render.
        custom_action (bool): parameter that determine whether to use custom action.
    """

    def __init__(
        self,
        name,
        mode,
        render=False,
        custom_action=False,
        **kwargs,
    ):
        self.env = gym.make(name)
        self.mode = mode
        self.state_size = self.env.observation_space.shape[0]
        if not custom_action:
            self.action_size = (
                self.env.action_space.shape[0]
                if mode == "continuous"
                else self.env.action_space.n
            )
        self.render = render

    def reset(self):
        self.score = 0
        state = self.env.reset()
        state = np.expand_dims(state, 0)  # for (1, state_size)
        return state

    def step(self, action):
        if self.render:
            self.env.render()
        if self.mode == "continuous":
            action = ((action + 1.0) / 2.0) * (
                self.env.action_space.high - self.env.action_space.low
            ) + self.env.action_space.low
            action = np.reshape(action, self.env.action_space.shape)
        else:
            action = np.asscalar(action)

        next_state, reward, done, info = self.env.step(action)
        self.score += reward

        next_state, reward, done = map(
            lambda x: np.expand_dims(x, 0), [next_state, [reward], [done]]
        )  # for (1, ?)
        return (next_state, reward, done)

    def close(self):
        self.env.close()


class Cartpole(_Gym):
    def __init__(self, mode="discrete", **kwargs):
        if mode == "continuous":
            super(Cartpole, self).__init__(
                "CartPole-v1", mode, custom_action=True, **kwargs
            )
            self.action_size = 1
        else:
            super(Cartpole, self).__init__("CartPole-v1", mode, **kwargs)

    def step(self, action):
        if self.render:
            self.env.render()
        action = np.asscalar(action)
        if self.mode == "continuous":
            action = 0 if action < 0 else 1
        next_state, reward, done, info = self.env.step(action)
        self.score += reward
        reward = -1 if done else 0.1

        next_state, reward, done = map(
            lambda x: np.expand_dims(x, 0), [next_state, [reward], [done]]
        )  # for (1, ?)
        return (next_state, reward, done)


class Pendulum(_Gym):
    def __init__(self, **kwargs):
        super(Pendulum, self).__init__("Pendulum-v0", "continuous", **kwargs)


class MountainCar(_Gym):
    def __init__(self, **kwargs):
        super(MountainCar, self).__init__("MountainCar-v0", "discrete", **kwargs)
