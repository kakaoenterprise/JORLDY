from .utils import check_interact, check_close, check_record
from core import Env
from core.env import env_dict


def test_nes(MockAgent):
    for name in [key for key, val in env_dict.items() if "procgen" in str(val)]:
        env = Env(name)

        agent = MockAgent(env.state_size, env.action_size, env.action_type)
        run_step = 10

        # test interact
        check_interact(env, agent, run_step)

        # test record
        check_record(env)

        # test close
        check_close(env)
