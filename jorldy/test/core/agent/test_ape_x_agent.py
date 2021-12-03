from core.agent.ape_x import ApeX
from .utils import check_interact, check_save_load, check_sync_in_out


def test_ape_x(MockEnv):
    state_size, action_size, action_type = 2, 3, "discrete"
    episode_len = 10
    env = MockEnv(state_size, action_size, action_type, episode_len)

    hidden_size = 4
    buffer_size, batch_size, start_train_step, target_update_period = 100, 4, 8, 5
    run_step = 20
    # ApeX
    epsilon, epsilon_alpha, clip_grad_norm = 0.4, 0.7, 40.0
    # PER
    alpha, beta = 0.6, 0.4
    learn_period, uniform_sample_prob = 4, 1e-3
    # Multistep
    n_step = 3
    agent = ApeX(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
        buffer_size=buffer_size,
        batch_size=batch_size,
        start_train_step=start_train_step,
        target_update_period=target_update_period,
        run_step=run_step,
        epsilon=epsilon,
        epsilon_alpha=epsilon_alpha,
        clip_grad_norm=clip_grad_norm,
        alpha=alpha,
        beta=beta,
        learn_period=learn_period,
        uniform_sample_prob=uniform_sample_prob,
        n_step=n_step,
    )

    # test after initialize
    assert agent.action_type == action_type

    # test inteact
    check_interact(env, agent, run_step)

    # test after inteact
    assert agent.time_t == run_step
    assert agent.memory.size == (run_step - n_step)
    assert agent.beta == 1.0

    # test save and load
    check_save_load(agent, "./tmp_test_ape_x")

    # test sync in and out
    check_sync_in_out(agent)
