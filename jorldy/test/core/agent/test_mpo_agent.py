from core.agent.mpo import MPO
from .utils import check_interact, check_save_load, check_sync_in_out


def test_discrete_retrace_mpo(MockEnv):
    state_size, action_size, action_type = 2, 3, "discrete"
    episode_len = 10
    env = MockEnv(state_size, action_size, action_type, episode_len)

    hidden_size = 4
    run_step = 20
    buffer_size, batch_size, start_train_step = 100, 4, 8
    use_standardization = True
    actor, critic = "discrete_policy", "discrete_q_network"
    n_step, n_epoch = 3, 2
    critic_loss_type = "retrace"
    num_sample = 10
    min_eta, min_alpha_mu, min_alpha_sigma = 1e-8, 1e-8, 1e-8
    eps_eta, eps_alpha_mu, eps_alpha_sigma = 0.01, 0.1, 5e-5
    eta, alpha_mu, alpha_sigma = 1.0, 1.0, 1.0
    agent = MPO(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
        actor=actor,
        critic=critic,
        critic_loss_type=critic_loss_type,
        buffer_size=buffer_size,
        batch_size=batch_size,
        start_train_step=start_train_step,
        use_standardization=use_standardization,
        run_step=run_step,
        n_step=n_step,
        n_epoch=n_epoch,
        num_sample=num_sample,
        min_eta=min_eta,
        min_alpha_mu=min_alpha_mu,
        min_alpha_sigma=min_alpha_sigma,
        eps_eta=eps_eta,
        eps_alpha_mu=eps_alpha_mu,
        eps_alpha_sigma=eps_alpha_sigma,
        eta=eta,
        alpha_mu=alpha_mu,
        alpha_sigma=alpha_sigma,
    )

    # test after initialize
    assert agent.action_type == action_type

    # test inteact
    check_interact(env, agent, run_step)

    # test after inteact

    # test save and load
    check_save_load(agent, "./tmp_test_discrete_retrace_mpo")

    # test sync in and out
    check_sync_in_out(agent)


def test_discrete_1step_TD_mpo(MockEnv):
    state_size, action_size, action_type = 2, 3, "discrete"
    episode_len = 10
    env = MockEnv(state_size, action_size, action_type, episode_len)

    hidden_size = 4
    run_step = 20
    buffer_size, batch_size, start_train_step = 100, 4, 8
    use_standardization = True
    actor, critic = "discrete_policy", "discrete_q_network"
    n_step, n_epoch = 3, 2
    critic_loss_type = "1step_TD"
    num_sample = 10
    min_eta, min_alpha_mu, min_alpha_sigma = 1e-8, 1e-8, 1e-8
    eps_eta, eps_alpha_mu, eps_alpha_sigma = 0.01, 0.1, 5e-5
    eta, alpha_mu, alpha_sigma = 1.0, 1.0, 1.0
    agent = MPO(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
        actor=actor,
        critic=critic,
        critic_loss_type=critic_loss_type,
        buffer_size=buffer_size,
        batch_size=batch_size,
        start_train_step=start_train_step,
        use_standardization=use_standardization,
        run_step=run_step,
        n_step=n_step,
        n_epoch=n_epoch,
        num_sample=num_sample,
        min_eta=min_eta,
        min_alpha_mu=min_alpha_mu,
        min_alpha_sigma=min_alpha_sigma,
        eps_eta=eps_eta,
        eps_alpha_mu=eps_alpha_mu,
        eps_alpha_sigma=eps_alpha_sigma,
        eta=eta,
        alpha_mu=alpha_mu,
        alpha_sigma=alpha_sigma,
    )

    # test after initialize
    assert agent.action_type == action_type

    # test inteact
    check_interact(env, agent, run_step)

    # test after inteact

    # test save and load
    check_save_load(agent, "./tmp_test_discrete_1step_TD_mpo")

    # test sync in and out
    check_sync_in_out(agent)


def test_continuous_retrace_mpo(MockEnv):
    state_size, action_size, action_type = 2, 3, "continuous"
    episode_len = 10
    env = MockEnv(state_size, action_size, action_type, episode_len)

    hidden_size = 4
    run_step = 20
    buffer_size, batch_size, start_train_step = 100, 4, 8
    use_standardization = True
    actor, critic = "continuous_policy", "continuous_q_network"
    n_step, n_epoch = 3, 2
    critic_loss_type = "retrace"
    num_sample = 10
    min_eta, min_alpha_mu, min_alpha_sigma = 1e-8, 1e-8, 1e-8
    eps_eta, eps_alpha_mu, eps_alpha_sigma = 0.01, 0.1, 5e-5
    eta, alpha_mu, alpha_sigma = 1.0, 1.0, 1.0
    agent = MPO(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
        actor=actor,
        critic=critic,
        critic_loss_type=critic_loss_type,
        buffer_size=buffer_size,
        batch_size=batch_size,
        start_train_step=start_train_step,
        use_standardization=use_standardization,
        run_step=run_step,
        n_step=n_step,
        n_epoch=n_epoch,
        num_sample=num_sample,
        min_eta=min_eta,
        min_alpha_mu=min_alpha_mu,
        min_alpha_sigma=min_alpha_sigma,
        eps_eta=eps_eta,
        eps_alpha_mu=eps_alpha_mu,
        eps_alpha_sigma=eps_alpha_sigma,
        eta=eta,
        alpha_mu=alpha_mu,
        alpha_sigma=alpha_sigma,
    )

    # test after initialize
    assert agent.action_type == action_type

    # test inteact
    check_interact(env, agent, run_step)

    # test after inteact

    # test save and load
    check_save_load(agent, "./tmp_test_continuous_retrace_mpo")

    # test sync in and out
    check_sync_in_out(agent)


def test_continuous_1step_TD_mpo(MockEnv):
    state_size, action_size, action_type = 2, 3, "continuous"
    episode_len = 10
    env = MockEnv(state_size, action_size, action_type, episode_len)

    hidden_size = 4
    run_step = 20
    buffer_size, batch_size, start_train_step = 100, 4, 8
    use_standardization = True
    actor, critic = "continuous_policy", "continuous_q_network"
    n_step, n_epoch = 3, 2
    critic_loss_type = "1step_TD"
    num_sample = 10
    min_eta, min_alpha_mu, min_alpha_sigma = 1e-8, 1e-8, 1e-8
    eps_eta, eps_alpha_mu, eps_alpha_sigma = 0.01, 0.1, 5e-5
    eta, alpha_mu, alpha_sigma = 1.0, 1.0, 1.0
    agent = MPO(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
        actor=actor,
        critic=critic,
        critic_loss_type=critic_loss_type,
        buffer_size=buffer_size,
        batch_size=batch_size,
        start_train_step=start_train_step,
        use_standardization=use_standardization,
        run_step=run_step,
        n_step=n_step,
        n_epoch=n_epoch,
        num_sample=num_sample,
        min_eta=min_eta,
        min_alpha_mu=min_alpha_mu,
        min_alpha_sigma=min_alpha_sigma,
        eps_eta=eps_eta,
        eps_alpha_mu=eps_alpha_mu,
        eps_alpha_sigma=eps_alpha_sigma,
        eta=eta,
        alpha_mu=alpha_mu,
        alpha_sigma=alpha_sigma,
    )

    # test after initialize
    assert agent.action_type == action_type

    # test inteact
    check_interact(env, agent, run_step)

    # test after inteact

    # test save and load
    check_save_load(agent, "./tmp_test_continuous_1step_TD_mpo")

    # test sync in and out
    check_sync_in_out(agent)
