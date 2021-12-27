from .utils import check_env
from core.env.mlagent import HopperMLAgent, PongMLAgent, DroneDeliveryMLAgent


def test_hopper_mlagent(MockAgent):
    env = HopperMLAgent()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_pong_mlagent(MockAgent):
    env = PongMLAgent()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)


def test_drone_delivery_mlagent(MockAgent):
    env = DroneDeliveryMLAgent()
    agent = MockAgent(env.state_size, env.action_size, env.action_type)

    check_env(env, agent)
