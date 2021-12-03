from core.agent.reinforce import REINFORCE
from .utils import check_interact, check_save_load, check_sync_in_out


def test_discrete_reinforce(MockEnv):
    state_size, action_size, action_type = 2, 3, "discrete"
    episode_len = 10
    env = MockEnv(state_size, action_size, action_type, episode_len)

    hidden_size = 4
    run_step = 25
    use_standardization = True
    network = "discrete_policy"
    agent = REINFORCE(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
        network=network,
        use_standardization=use_standardization,
        run_step=run_step,
    )

    # test after initialize
    assert agent.action_type == action_type

    # test inteact
    check_interact(env, agent, run_step)

    # test after inteact
    assert agent.memory.size == (run_step % episode_len)

    # test save and load
    check_save_load(agent, "./tmp_test_discrete_reinforce")

    # test sync in and out
    check_sync_in_out(agent)


def test_continuous_reinforce(MockEnv):
    state_size, action_size, action_type = 2, 3, "continuous"
    episode_len = 10
    env = MockEnv(state_size, action_size, action_type, episode_len)

    hidden_size = 4
    run_step = 25
    use_standardization = True
    network = "continuous_policy"
    agent = REINFORCE(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
        network=network,
        use_standardization=use_standardization,
        run_step=run_step,
    )

    # test after initialize
    assert agent.action_type == action_type

    # test inteact
    check_interact(env, agent, run_step)

    # test after inteact
    assert agent.memory.size == (run_step % episode_len)

    # test save and load
    check_save_load(agent, "./tmp_test_continuous_reinforce")

    # test sync in and out
    check_sync_in_out(agent)
