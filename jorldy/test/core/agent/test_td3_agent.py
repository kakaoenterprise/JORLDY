from core.agent.td3 import TD3
from .utils import check_interact, check_save_load, check_sync_in_out


def test_td3(MockEnv):
    state_size, action_size, action_type = 2, 3, "continuous"
    episode_len = 10
    env = MockEnv(state_size, action_size, action_type, episode_len)

    hidden_size = 4
    buffer_size, batch_size, start_train_step = 100, 4, 8
    run_step = 20
    tau = 1e-3
    mu, theta, sigma = 0, 1e-3, 2e-3
    agent = TD3(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
        buffer_size=buffer_size,
        batch_size=batch_size,
        start_train_step=start_train_step,
        run_step=run_step,
        tau=tau,
        mu=mu,
        theta=theta,
        sigma=sigma,
    )

    # test after initialize
    assert agent.action_type == action_type

    # test inteact
    check_interact(env, agent, run_step)

    # test after inteact
    assert agent.num_learn == (run_step - start_train_step + 1)

    # test save and load
    check_save_load(agent, "./tmp_test_td3")

    # test sync in and out
    check_sync_in_out(agent)
