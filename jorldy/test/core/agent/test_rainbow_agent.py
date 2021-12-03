from core.agent.rainbow import Rainbow
from .utils import check_interact, check_save_load, check_sync_in_out


def test_rainbow(MockEnv):
    state_size, action_size, action_type = 2, 3, "discrete"
    episode_len = 10
    env = MockEnv(state_size, action_size, action_type, episode_len)

    hidden_size = 4
    buffer_size, batch_size, start_train_step, target_update_period = 100, 4, 8, 5
    run_step = 20
    # MultiStep
    n_step = 3
    # PER
    alpha, beta = 0.6, 0.4
    learn_period, uniform_sample_prob = 4, 1e-3
    # Noisy
    noise_type = "factorized"  # [independent, factorized]
    # C51
    v_min, v_max, num_support = -5, 5, 5
    agent = Rainbow(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
        buffer_size=buffer_size,
        batch_size=batch_size,
        start_train_step=start_train_step,
        target_update_period=target_update_period,
        run_step=run_step,
        n_step=n_step,
        alpha=alpha,
        beta=beta,
        learn_period=learn_period,
        uniform_sample_prob=uniform_sample_prob,
        noise_type=noise_type,
        v_min=v_min,
        v_max=v_max,
        num_support=num_support,
    )

    # test after initialize
    assert agent.action_type == action_type
    assert agent.z.shape == (1, num_support)

    # test inteact
    check_interact(env, agent, run_step)

    # test after inteact
    assert agent.time_t == run_step
    assert agent.beta == 1.0

    # test save and load
    check_save_load(agent, "./tmp_test_rainbow")

    # test sync in and out
    check_sync_in_out(agent)
