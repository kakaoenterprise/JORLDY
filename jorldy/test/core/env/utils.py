def check_interact(env, agent, run_step):
    state = env.reset()
    for _ in range(run_step):
        action_dict = agent.act(state)
        next_state, reward, done = env.step(action_dict["action"])

        if isinstance(env.state_size, int):
            assert next_state.shape == (1, env.state_size)
        elif isinstance(env.state_size, list):
            if isinstance(env.state_size[0], list):
                assert next_state[0].shape == (1, *env.state_size[0])
                assert next_state[1].shape == (1, env.state_size[1])
            else:
                assert next_state.shape == (1, *env.state_size)
        assert reward.shape == (1, 1)
        assert done.shape == (1, 1)

        state = env.reset() if done else next_state


def check_close(env):
    env.close()


def check_record(env):
    if env.recordable():
        env.get_frame()


def check_env(env, agent, run_step=10):

    # test interact
    check_interact(env, agent, run_step)

    # test record
    check_record(env)

    # test close
    check_close(env)
