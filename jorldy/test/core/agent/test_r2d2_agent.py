from core.agent.r2d2 import R2D2
from .utils import check_interact, check_save_load, check_sync_in_out


def test_r2d2(MockEnv):
    state_size, action_size, action_type = 2, 3, "discrete"
    episode_len = 10
    env = MockEnv(state_size, action_size, action_type, episode_len)

    hidden_size = 4
    buffer_size, batch_size, start_train_step, target_update_period = 100, 4, 8, 5
    run_step = 20
    # APE-X
    epsilon, epsilon_alpha, clip_grad_norm = 0.4, 0.7, 40.0
    # PER
    alpha, beta = 0.6, 0.4
    learn_period, uniform_sample_prob = 4, 1e-3
    # Multistep
    n_step = 5
    # R2D2
    seq_len, n_burn_in, zero_padding = 10, 2, True

    agent = R2D2(
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
        seq_len=seq_len,
        n_burn_in=n_burn_in,
        zero_padding=zero_padding,
    )

    # test after initialize
    assert agent.action_type == action_type

    # test inteact
    check_interact(env, agent, run_step)

    # test after inteact
    last_episode_len = (
        episode_len if run_step % episode_len == 0 else run_step % episode_len
    )

    assert (
        agent.time_t
        == run_step - max(0, (last_episode_len - n_step - 1)) % agent.store_period
    )
    assert agent.store_period == seq_len // 2
    assert agent.memory.size == ((run_step - n_step) // episode_len) * (
        (episode_len // agent.store_period) + bool(episode_len % agent.store_period)
    ) + (((run_step - n_step) % episode_len) // agent.store_period) + bool(
        ((run_step - n_step) % episode_len) % agent.store_period
    )

    # test save and load
    check_save_load(agent, "./tmp_test_r2d2")

    # test sync in and out
    check_sync_in_out(agent)
