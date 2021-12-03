def check_interact(env, agent, run_step):
    state = env.reset()
    for _ in range(run_step):
        action_dict = agent.act(state)
        next_state, reward, done = env.step(action_dict["action"])

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
