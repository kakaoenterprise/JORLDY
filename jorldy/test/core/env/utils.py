import numpy as np


class MockAgent:
    def __init__(self, state_size, action_size, action_type):
        self.state_size = state_size
        self.action_size = action_size
        self.action_type = action_type

        assert self.action_type in ["discrete", "continuous"]

    def act(self, state):
        batch_size = state[0].shape[0] if isinstance(state, list) else state.shape[0]

        if self.action_type == "discrete":
            action = np.random.randint(0, self.action_size, size=(batch_size, 1))
        else:
            action = np.random.random((batch_size, self.action_size))

        return action


def check_interact(env, agent, run_step):
    state = env.reset()
    for _ in range(run_step):
        action = agent.act(state)
        next_state, reward, done = env.step(action)

        if isinstance(env.state_size, int):
            assert next_state.shape == (1, env.state_size)
        elif isinstance(env.state_size, list):
            assert next_state.shape == (1, *env.state_size)
        assert reward.shape == (1, 1)
        assert done.shape == (1, 1)

        state = env.reset() if done else next_state


def check_close(env):
    env.close()


def check_record(env):
    if env.recordable():
        env.get_frame()
