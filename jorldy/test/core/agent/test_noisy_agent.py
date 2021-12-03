from core.agent.noisy import Noisy
from .utils import check_interact, check_save_load, check_sync_in_out


def test_noisy(MockEnv):
    state_size, action_size, action_type = 2, 3, "discrete"
    episode_len = 10
    env = MockEnv(state_size, action_size, action_type, episode_len)

    hidden_size = 4
    buffer_size, batch_size, start_train_step, target_update_period = 100, 4, 8, 5
    run_step = 20
    agent = Noisy(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
        buffer_size=buffer_size,
        batch_size=batch_size,
        start_train_step=start_train_step,
        target_update_period=target_update_period,
        run_step=run_step,
    )

    # test after initialize
    assert agent.action_type == action_type

    # test inteact
    check_interact(env, agent, run_step)

    # test after inteact
    assert agent.time_t == run_step

    # test save and load
    check_save_load(agent, "./tmp_test_noisy")

    # test sync in and out
    check_sync_in_out(agent)
