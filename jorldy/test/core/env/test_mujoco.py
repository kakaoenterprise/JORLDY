import sys
import pytest

if sys.platform.startswith("win"):
    pytest.skip("mujoco is not supported in windows", allow_module_level=True)

from .utils import check_env
from core.env.mujoco import (
    HalfCheetah,
    Ant,
    Hopper,
    Humanoid,
    HumanoidStandup,
    InvertedDoublePendulum,
    InvertedPendulum,
    Reacher,
    Swimmer,
    Walker,
)


def test_half_cheetah(MockAgent):
    env = HalfCheetah()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_ant(MockAgent):
    env = Ant()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_hopper(MockAgent):
    env = Hopper()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_humanoid(MockAgent):
    env = Humanoid()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_humanoid_standup(MockAgent):
    env = HumanoidStandup()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_invert_double_pendulum(MockAgent):
    env = InvertedDoublePendulum()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_inverted_pendulum(MockAgent):
    env = InvertedPendulum()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_reacher(MockAgent):
    env = Reacher()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_swimmer(MockAgent):
    env = Swimmer()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_walker(MockAgent):
    env = Walker()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)
