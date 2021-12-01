import gym
import numpy as np
from .base import BaseEnv


class _Gym(BaseEnv):
    """Gym environment.

    Args:
        name (str): name of environment in Gym.
        render (bool): parameter that determine whether to render.
        custom_action (bool): parameter that determine whether to use custom action.
    """

    def __init__(
        self,
        name,
        render=False,
        custom_action=False,
        **kwargs,
    ):
        self.env = gym.make(name)
        self.state_size = self.env.observation_space.shape[0]
        if not custom_action:
            self.action_size = (
                self.env.action_space.shape[0]
                if self.action_type == "continuous"
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
        if self.action_type == "continuous":
            action = ((action + 1.0) / 2.0) * (
                self.env.action_space.high - self.env.action_space.low
            ) + self.env.action_space.low
            action = np.reshape(action, self.env.action_space.shape)
        else:
            action = action.item()

        next_state, reward, done, info = self.env.step(action)
        self.score += reward

        next_state, reward, done = map(
            lambda x: np.expand_dims(x, 0), [next_state, [reward], [done]]
        )  # for (1, ?)
        return (next_state, reward, done)

    def close(self):
        self.env.close()


class Cartpole(_Gym):
    def __init__(self, action_type="discrete", **kwargs):
        self.action_type = action_type
        if action_type == "continuous":
            super(Cartpole, self).__init__("CartPole-v1", custom_action=True, **kwargs)
            self.action_size = 1
        else:
            super(Cartpole, self).__init__("CartPole-v1", **kwargs)

    def step(self, action):
        if self.render:
            self.env.render()
        action = action.item()
        if self.action_type == "continuous":
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
        self.action_type = "continuous"
        super(Pendulum, self).__init__("Pendulum-v1", **kwargs)


class MountainCar(_Gym):
    def __init__(self, **kwargs):
        self.action_type = "discrete"
        super(MountainCar, self).__init__("MountainCar-v0", **kwargs)
