from core.agent.rnd_ppo import RND_PPO
from .utils import check_interact, check_save_load, check_sync_in_out


def test_discrete_rnd_ppo(MockEnv):
    state_size, action_size, action_type = 2, 3, "discrete"
    episode_len = 10
    env = MockEnv(state_size, action_size, action_type, episode_len)

    hidden_size, batch_size = 4, 4
    run_step = 20
    use_standardization = True
    network = "discrete_policy_separate_value"
    n_step, n_epoch = 8, 2
    gamma_i = 0.99
    extrinsic_coeff, intrinsic_coeff = (
        1.0,
        1.0,
    )
    obs_normalize, ri_normalize, batch_norm = True, True, True
    agent = RND_PPO(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
        network=network,
        batch_size=batch_size,
        use_standardization=use_standardization,
        run_step=run_step,
        n_step=n_step,
        n_epoch=n_epoch,
        gamma_i=gamma_i,
        extrinsic_coeff=extrinsic_coeff,
        intrinsic_coeff=intrinsic_coeff,
        obs_normalize=obs_normalize,
        ri_normalize=ri_normalize,
        batch_norm=batch_norm,
    )

    # test after initialize
    assert agent.action_type == action_type

    # test inteact
    check_interact(env, agent, run_step)

    # test after inteact
    assert agent.memory.size == (run_step % n_step)

    # test save and load
    check_save_load(agent, "./tmp_test_discrete_rnd_ppo")

    # test sync in and out
    check_sync_in_out(agent)


def test_continuous_rnd_ppo(MockEnv):
    state_size, action_size, action_type = 2, 3, "continuous"
    episode_len = 10
    env = MockEnv(state_size, action_size, action_type, episode_len)

    hidden_size, batch_size = 4, 4
    run_step = 20
    use_standardization = True
    network = "continuous_policy_separate_value"
    n_step, n_epoch = 8, 2
    gamma_i = 0.99
    extrinsic_coeff, intrinsic_coeff = (
        1.0,
        1.0,
    )
    obs_normalize, ri_normalize, batch_norm = True, True, True
    agent = RND_PPO(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
        network=network,
        batch_size=batch_size,
        use_standardization=use_standardization,
        run_step=run_step,
        n_step=n_step,
        n_epoch=n_epoch,
        gamma_i=gamma_i,
        extrinsic_coeff=extrinsic_coeff,
        intrinsic_coeff=intrinsic_coeff,
        obs_normalize=obs_normalize,
        ri_normalize=ri_normalize,
        batch_norm=batch_norm,
    )

    # test after initialize
    assert agent.action_type == action_type

    # test inteact
    check_interact(env, agent, run_step)

    # test after inteact
    assert agent.memory.size == (run_step % n_step)

    # test save and load
    check_save_load(agent, "./tmp_test_continuous_rnd_ppo")

    # test sync in and out
    check_sync_in_out(agent)
