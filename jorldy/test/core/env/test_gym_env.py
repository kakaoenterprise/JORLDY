from .utils import check_interact, check_close, check_record
from core import Env


def test_discrete_cartpole(MockAgent):
    name = "cartpole"
    action_type = "discrete"
    env = Env(name, action_type)

    agent = MockAgent(env.state_size, env.action_size, env.action_type)
    run_step = 10

    # test interact
    check_interact(env, agent, run_step)

    # test record
    check_record(env)

    # test close
    check_close(env)


def test_continuous_cartpole(MockAgent):
    name = "cartpole"
    action_type = "continuous"
    env = Env(name, action_type)

    agent = MockAgent(env.state_size, env.action_size, env.action_type)
    run_step = 10

    # test interact
    check_interact(env, agent, run_step)

    # test record
    check_record(env)

    # test close
    check_close(env)


def test_pendulum(MockAgent):
    name = "pendulum"
    env = Env(name)

    agent = MockAgent(env.state_size, env.action_size, env.action_type)
    run_step = 10

    # test interact
    check_interact(env, agent, run_step)

    # test record
    check_record(env)

    # test close
    check_close(env)


def test_mountain_car(MockAgent):
    name = "pendulum"
    env = Env(name)

    agent = MockAgent(env.state_size, env.action_size, env.action_type)
    run_step = 10

    # test interact
    check_interact(env, agent, run_step)

    # test record
    check_record(env)

    # test close
    check_close(env)
