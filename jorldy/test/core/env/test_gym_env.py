from .utils import check_env
from core.env.gym_env import Cartpole, Pendulum, MountainCar


def test_discrete_cartpole(MockAgent):
    action_type = "discrete"
    env = Cartpole(action_type=action_type)
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_continuous_cartpole(MockAgent):
    action_type = "discrete"
    env = Cartpole(action_type=action_type)
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_pendulum(MockAgent):
    env = Pendulum()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_mountain_car(MockAgent):
    env = MountainCar()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)
