from core.agent.m_iqn import M_IQN
from .utils import check_interact, check_save_load, check_sync_in_out


def test_m_iqn(MockEnv):
    state_size, action_size, action_type = 2, 3, "discrete"
    episode_len = 10
    env = MockEnv(state_size, action_size, action_type, episode_len)

    hidden_size = 4
    epsilon_init, epsilon_min, explore_ratio = 1.0, 0.1, 0.2
    buffer_size, batch_size, start_train_step, target_update_period = 100, 4, 8, 5
    run_step = 20
    num_sample, embedding_dim, sample_min, sample_max = 8, 16, 0.0, 1.0
    # M-IQN parameters
    alpha, tau, l_0 = 0.9, 0.03, -1
    agent = M_IQN(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
        epsilon_init=epsilon_init,
        epsilon_min=epsilon_min,
        explore_ratio=explore_ratio,
        buffer_size=buffer_size,
        batch_size=batch_size,
        start_train_step=start_train_step,
        target_update_period=target_update_period,
        run_step=run_step,
        num_sample=num_sample,
        embedding_dim=embedding_dim,
        sample_min=sample_min,
        sample_max=sample_max,
        alpha=alpha,
        tau=tau,
        l_0=l_0,
    )

    # test after initialize
    assert agent.action_type == action_type
    assert agent.epsilon == epsilon_init

    # test inteact
    check_interact(env, agent, run_step)

    # test after inteact
    assert agent.epsilon == epsilon_min
    assert agent.time_t == run_step

    # test save and load
    check_save_load(agent, "./tmp_test_m_iqn")

    # test sync in and out
    check_sync_in_out(agent)
