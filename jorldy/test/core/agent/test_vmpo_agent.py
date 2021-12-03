from core.agent.vmpo import VMPO
from .utils import check_interact, check_save_load, check_sync_in_out


def test_discrete_vmpo(MockEnv):
    state_size, action_size, action_type = 2, 3, "discrete"
    episode_len = 10
    env = MockEnv(state_size, action_size, action_type, episode_len)

    hidden_size, batch_size = 4, 4
    run_step = 20
    use_standardization = True
    network = "discrete_policy_value"
    n_step, n_epoch = 8, 2
    min_eta, min_alpha_mu, min_alpha_sigma = 1e-8, 1e-8, 1e-8
    eps_eta, eps_alpha_mu, eps_alpha_sigma = 0.02, 0.1, 0.1
    eta, alpha_mu, alpha_sigma = 1.0, 1.0, 1.0
    agent = VMPO(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
        network=network,
        batch_size=batch_size,
        use_standardization=use_standardization,
        run_step=run_step,
        n_step=n_step,
        n_epoch=n_epoch,
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
    assert agent.memory.size == (run_step % n_step)

    # test save and load
    check_save_load(agent, "./tmp_test_discrete_vmpo")

    # test sync in and out
    check_sync_in_out(agent)


def test_continuous_vmpo(MockEnv):
    state_size, action_size, action_type = 2, 3, "continuous"
    episode_len = 10
    env = MockEnv(state_size, action_size, action_type, episode_len)

    hidden_size, batch_size = 4, 4
    run_step = 20
    use_standardization = True
    network = "continuous_policy_value"
    n_step, n_epoch = 8, 2
    agent = VMPO(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
        network=network,
        batch_size=batch_size,
        use_standardization=use_standardization,
        run_step=run_step,
        n_step=n_step,
        n_epoch=n_epoch,
    )

    # test after initialize
    assert agent.action_type == action_type

    # test inteact
    check_interact(env, agent, run_step)

    # test after inteact
    assert agent.memory.size == (run_step % n_step)

    # test save and load
    check_save_load(agent, "./tmp_test_continuous_vmpo")

    # test sync in and out
    check_sync_in_out(agent)
