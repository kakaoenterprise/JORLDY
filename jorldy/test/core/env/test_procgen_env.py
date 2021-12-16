from .utils import check_env
from core.env.procgen import (
    Coinrun,
    Bigfish,
    Bossfight,
    Caveflyer,
    Chaser,
    Climber,
    Dodgeball,
    Fruitbot,
    Heist,
    Jumper,
    Leaper,
    Maze,
    Miner,
    Ninja,
    Plunder,
    Starpilot,
)


def test_coinrun(MockAgent):
    env = Coinrun()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_bigfish(MockAgent):
    env = Bigfish()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_bossfight(MockAgent):
    env = Bossfight()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_caveflyer(MockAgent):
    env = Caveflyer()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_chaser(MockAgent):
    env = Chaser()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_climber(MockAgent):
    env = Climber()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_dodgeball(MockAgent):
    env = Dodgeball()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_fruitbot(MockAgent):
    env = Fruitbot()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_heist(MockAgent):
    env = Heist()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_jumper(MockAgent):
    env = Jumper()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_leaper(MockAgent):
    env = Leaper()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_maze(MockAgent):
    env = Maze()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_miner(MockAgent):
    env = Miner()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_ninja(MockAgent):
    env = Ninja()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_plunder(MockAgent):
    env = Plunder()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_starpilot(MockAgent):
    env = Starpilot()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)
