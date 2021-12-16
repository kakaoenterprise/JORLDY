from .utils import check_env
from core.env.atari import (
    Breakout,
    Pong,
    Asterix,
    Assault,
    Seaquest,
    Spaceinvaders,
    Alien,
    CrazyClimber,
    Enduro,
    Qbert,
    PrivateEye,
    MontezumaRevenge,
)


def test_breakout(MockAgent):
    env = Breakout()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_pong(MockAgent):
    env = Pong()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_asterix(MockAgent):
    env = Asterix()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_assault(MockAgent):
    env = Assault()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_seaquest(MockAgent):
    env = Seaquest()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_spaceinvaders(MockAgent):
    env = Spaceinvaders()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_alien(MockAgent):
    env = Alien()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_carzy_climber(MockAgent):
    env = CrazyClimber()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_enduro(MockAgent):
    env = Enduro()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_qbert(MockAgent):
    env = Qbert()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_private_eye(MockAgent):
    env = PrivateEye()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_montezuma_revenge(MockAgent):
    env = MontezumaRevenge()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)
