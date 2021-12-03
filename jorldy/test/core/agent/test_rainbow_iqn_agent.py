from core.agent.rainbow_iqn import RainbowIQN
from .utils import check_interact, check_save_load, check_sync_in_out


def test_rainbow_iqn(MockEnv):
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
    # IQN
    num_sample, embedding_dim, sample_min, sample_max = 8, 16, 0.0, 1.0
    agent = RainbowIQN(
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
        num_sample=num_sample,
        embedding_dim=embedding_dim,
        sample_min=sample_min,
        sample_max=sample_max,
    )

    # test after initialize
    assert agent.action_type == action_type

    # test inteact
    check_interact(env, agent, run_step)

    # test after inteact
    assert agent.time_t == run_step
    assert agent.beta == 1.0

    # test save and load
    check_save_load(agent, "./tmp_test_rainbow_iqn")

    # test sync in and out
    check_sync_in_out(agent)
