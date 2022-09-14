from core.agent.muzero import Muzero
from .utils import check_interact, check_save_load, check_sync_in_out


def test_muzero(MockEnv):
    state_size, action_size, action_type = 2, 3, "discrete"
    episode_len = 10
    env = MockEnv(state_size, action_size, action_type, episode_len)

    hidden_size, buffer_size, batch_size, start_train_step = 4, 100, 4, 8
    run_step = 20
    # MuZero
    num_stack, num_unroll, num_td_step = 4, 3, 2
    num_simulation, value_loss_weight, trajectory_size = 5, 0.25, 10
    # PER
    alpha, beta = 0.6, 0.4
    learn_period, uniform_sample_prob = 4, 1e-3

    agent = Muzero(
        state_size=state_size,
        hidden_size=hidden_size,
        action_size=action_size,
        batch_size=batch_size,
        start_train_step=start_train_step,
        max_trajectory_size=trajectory_size,
        value_loss_weight=value_loss_weight,
        num_simulation=num_simulation,
        num_unroll=num_unroll,
        num_td_step=num_td_step,
        num_stack=num_stack,
        buffer_size=buffer_size,
        run_step=run_step,
        alpha=alpha,
        beta=beta,
        learn_period=learn_period,
        uniform_sample_prob=uniform_sample_prob,
    )

    # test after initialize
    assert agent.action_type == action_type

    # test inteact
    check_interact(env, agent, run_step)

    # test after inteact
    last_episode_len = (
        episode_len if run_step % episode_len == 0 else run_step % episode_len
    )

    assert agent.time_t == run_step

    # test save and load
    check_save_load(agent, "./tmp_test_muzero")

    # test sync in and out
    check_sync_in_out(agent)
