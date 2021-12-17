from .utils import check_env
from core.env.mujoco import HalfCheetah, Ant, Hopper, Humanoid, HumanoidStandup, InvertedDoublePendulum, InvertedPendulum, Reacher, Swimmer, Walker
import platform 

os = platform.system()

def test_half_cheetah(MockAgent):
    if os == 'Darwin' or 'Linux':
        env = HalfCheetah()
        agent = MockAgent(env.state_size, env.action_size, env.action_type)

        check_env(env, agent)


def test_ant(MockAgent):
    if os == 'Darwin' or 'Linux':
        env = Ant()
        agent = MockAgent(env.state_size, env.action_size, env.action_type)

        check_env(env, agent)

def test_hopper(MockAgent):
    if os == 'Darwin' or 'Linux':
        env = Hopper()
        agent = MockAgent(env.state_size, env.action_size, env.action_type)

        check_env(env, agent)

def test_humanoid(MockAgent):
    if os == 'Darwin' or 'Linux':
        env = Humanoid()
        agent = MockAgent(env.state_size, env.action_size, env.action_type)

        check_env(env, agent)

def test_humanoid_standup(MockAgent):
    if os == 'Darwin' or 'Linux':
        env = HumanoidStandup()
        agent = MockAgent(env.state_size, env.action_size, env.action_type)

        check_env(env, agent)

def test_invert_double_pendulum(MockAgent):
    if os == 'Darwin' or 'Linux':
        env = InvertedDoublePendulum()
        agent = MockAgent(env.state_size, env.action_size, env.action_type)

        check_env(env, agent)

def test_inverted_pendulum(MockAgent):
    if os == 'Darwin' or 'Linux':
        env = InvertedPendulum()
        agent = MockAgent(env.state_size, env.action_size, env.action_type)

        check_env(env, agent)

def test_reacher(MockAgent):
    if os == 'Darwin' or 'Linux':
        env = Reacher()
        agent = MockAgent(env.state_size, env.action_size, env.action_type)

        check_env(env, agent)

def test_swimmer(MockAgent):
    if os == 'Darwin' or 'Linux':
        env = Swimmer()
        agent = MockAgent(env.state_size, env.action_size, env.action_type)

        check_env(env, agent)

def test_walker(MockAgent):
    if os == 'Darwin' or 'Linux':
        env = Walker()
        agent = MockAgent(env.state_size, env.action_size, env.action_type)

        check_env(env, agent)
