from core.agent.sac import SAC
from .utils import check_interact, check_save_load, check_sync_in_out


def test_continuous_sac_dynamic_alpha_sac(MockEnv):
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
        actor="continuous_policy",
        critic="continuous_q_network",
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
    check_interact(env, agent, run_step)

    # test after inteact
    assert agent.num_learn == (run_step - start_train_step + 1)

    # test save and load
    check_save_load(agent, "./tmp_test_dynamic_alpha_sac")

    # test sync in and out
    check_sync_in_out(agent)


def test_continuous_sac_static_alpha_sac(MockEnv):
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
        actor="continuous_policy",
        critic="continuous_q_network",
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
    check_interact(env, agent, run_step)

    # test after inteact
    assert agent.num_learn == (run_step - start_train_step + 1)

    # test save and load
    check_save_load(agent, "./tmp_test_static_alpha_sac")

    # test sync in and out
    check_sync_in_out(agent)


def test_discrete_sac(MockEnv):
    state_size, action_size, action_type = 2, 3, "discrete"
    episode_len = 10
    env = MockEnv(state_size, action_size, action_type, episode_len)

    hidden_size = 4
    buffer_size, batch_size, start_train_step = 100, 4, 8
    run_step = 20
    use_dynamic_alpha = True
    static_log_alpha = -2.0
    tau = 5e-3
    target_update_period = 5
    agent = SAC(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
        actor="discrete_policy",
        critic="discrete_q_network",
        buffer_size=buffer_size,
        batch_size=batch_size,
        start_train_step=start_train_step,
        run_step=run_step,
        use_dynamic_alpha=use_dynamic_alpha,
        static_log_alpha=static_log_alpha,
        tau=tau,
        target_update_period=target_update_period,
    )

    # test after initialize
    assert agent.action_type == action_type

    # test inteact
    check_interact(env, agent, run_step)

    # test after inteact
    assert agent.num_learn == (run_step - start_train_step + 1)

    # test save and load
    check_save_load(agent, "./tmp_test_discrete_sac")

    # test sync in and out
    check_sync_in_out(agent)
