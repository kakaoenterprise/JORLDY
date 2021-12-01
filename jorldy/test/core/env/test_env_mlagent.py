from .utils import MockAgent, check_interact, check_close, check_record
from core import Env
from core.env import env_dict


def test_atari():
    for name in [key for key, val in env_dict.items() if "mlagent" in str(val)]:
        if "drone" in name:
            continue
        env = Env(name)

        agent = MockAgent(env.state_size, env.action_size, env.action_type)
        run_step = 10

        # test interact
        check_interact(env, agent, run_step)

        # test close
        check_close(env)

        # test record
        check_record(env)
