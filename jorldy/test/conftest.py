import os, sys

sys.path.append(os.getcwd())  # pytest should be run on JORLDY/jorldy

import numpy as np
import pytest


class _MockEnv:
    def __init__(self, state_size, action_size, action_type, episode_len, **kwargs):
        self.state_size = state_size
        self.action_size = action_size
        self.action_type = action_type
        self.episode_len = episode_len
        self.time_t = 0

    def reset(self):
        self.score = 0
        state = (
            np.random.random((1, self.state_size))
            if isinstance(self.state_size, int)
            else np.random.randint(low=0, high=255, size=self.state_size)
        )
        return state

    def step(self, action):
        self.time_t += 1
        next_state = (
            np.random.random((1, self.state_size))
            if isinstance(self.state_size, int)
            else np.random.randint(low=0, high=255, size=self.state_size)
        )
        reward = np.random.random((1, 1))
        done = np.array([[self.time_t == self.episode_len]])
        if done:
            self.time_t = 0

        self.score += 1
        return next_state, reward, done

    def get_frame(self):
        return np.random.randint(low=0, high=255, size=self.state_size)

    def recordable(self):
        return True


class _MockAgent:
    def __init__(self, state_size, action_size, action_type, **kwargs):
        self.state_size = state_size
        self.action_size = action_size
        self.action_type = action_type

        assert self.action_type in ["discrete", "continuous"]

    def act(self, state, training=True):
        batch_size = state[0].shape[0] if isinstance(state, list) else state.shape[0]

        if self.action_type == "discrete":
            action = np.random.randint(0, self.action_size, size=(batch_size, 1))
        else:
            action = np.random.random((batch_size, self.action_size))

        return {"action": action}

    def interact_callback(self, transition):
        return transition


_mock_transition = [
    {
        "state": np.random.random((1, 4)),
        "action": np.random.random((1, 3)),
        "reward": np.random.random((1, 1)),
        "next_state": np.random.random((1, 4)),
        "done": np.random.random((1, 1)) < 0.5,
        "multi_modal": [np.random.random((1, 3, 8, 8)), np.random.random((1, 4))],
        "seq": np.random.random((1, 3, 4)),
    },
]

_env_config = {
    "state_size": [3, 32, 32],
    "action_size": 2,
    "action_type": "discrete",
    "episode_len": 5,
}

_agent_config = {
    "state_size": [3, 32, 32],
    "action_size": 2,
    "action_type": "discrete",
}


@pytest.fixture
def MockEnv():
    return _MockEnv


@pytest.fixture
def MockAgent():
    return _MockAgent


@pytest.fixture
def mock_transition():
    return _mock_transition


@pytest.fixture
def env_config():
    return _env_config


@pytest.fixture
def agent_config():
    return _agent_config
