from core.agent.sac import SAC
from utils import MockEnv, check_interact, check_save_load, check_sync_in_out


def test_dynamic_alpha_sac():
    state_size, action_size, action_type = 2, 3, "continuous"
    episode_len = 10
    env = MockEnv(state_size, action_size, action_type, episode_len)

    hidden_size = 4
    buffer_size, batch_size, start_train_step = 100, 4, 8
    run_step = 20
    use_dynamic_alpha = True
    tau = 5e-3
    agent = SAC(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
        buffer_size=buffer_size,
        batch_size=batch_size,
        start_train_step=start_train_step,
        run_step=run_step,
        use_dynamic_alpha=use_dynamic_alpha,
        tau=tau,
    )

    # test after initialize
    assert agent.action_type == action_type

    # test inteact
    action_branch = 1 if action_type == "discrete" else action_size
    check_interact(env, agent, run_step, action_branch)

    # test after inteact
    assert agent.num_learn == (run_step - start_train_step + 1)

    # test save and load
    check_save_load(agent, "./tmp_test_dynamic_alpha_sac")

    # sync in and out
    check_sync_in_out(agent)


def test_static_alpha_sac():
    state_size, action_size, action_type = 2, 3, "continuous"
    episode_len = 10
    env = MockEnv(state_size, action_size, action_type, episode_len)

    hidden_size = 4
    buffer_size, batch_size, start_train_step = 100, 4, 8
    run_step = 20
    use_dynamic_alpha = False
    static_log_alpha = -2.0
    tau = 5e-3
    agent = SAC(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
        buffer_size=buffer_size,
        batch_size=batch_size,
        start_train_step=start_train_step,
        run_step=run_step,
        use_dynamic_alpha=use_dynamic_alpha,
        static_log_alpha=static_log_alpha,
        tau=tau,
    )

    # test after initialize
    assert agent.action_type == action_type

    # test inteact
    action_branch = 1 if action_type == "discrete" else action_size
    check_interact(env, agent, run_step, action_branch)

    # test after inteact
    assert agent.num_learn == (run_step - start_train_step + 1)

    # test save and load
    check_save_load(agent, "./tmp_test_static_alpha_sac")

    # sync in and out
    check_sync_in_out(agent)
