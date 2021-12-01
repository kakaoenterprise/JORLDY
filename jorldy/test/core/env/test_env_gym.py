from .utils import MockAgent, check_interact, check_close, check_record
from core import Env


def test_discrete_cartpole():
    name = "cartpole"
    action_type = "discrete"
    env = Env(name, action_type)

    agent = MockAgent(env.state_size, env.action_size, env.action_type)
    run_step = 10

    # test interact
    check_interact(env, agent, run_step)

    # test close
    check_close(env)

    # test record
    check_record(env)


def test_continuous_cartpole():
    name = "cartpole"
    action_type = "continuous"
    env = Env(name, action_type)

    agent = MockAgent(env.state_size, env.action_size, env.action_type)
    run_step = 10

    # test interact
    check_interact(env, agent, run_step)

    # test close
    check_close(env)

    # test record
    check_record(env)


def test_pendulum():
    name = "pendulum"
    env = Env(name)

    agent = MockAgent(env.state_size, env.action_size, env.action_type)
    run_step = 10

    # test interact
    check_interact(env, agent, run_step)

    # test close
    check_close(env)

    # test record
    check_record(env)


def test_mountain_car():
    name = "pendulum"
    env = Env(name)

    agent = MockAgent(env.state_size, env.action_size, env.action_type)
    run_step = 10

    # test interact
    check_interact(env, agent, run_step)

    # test close
    check_close(env)

    # test record
    check_record(env)
