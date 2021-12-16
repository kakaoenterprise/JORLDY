from .utils import check_env
from core.env.nes import SuperMarioBros


def test_super_mario_bros(MockAgent):
    env = SuperMarioBros()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)
