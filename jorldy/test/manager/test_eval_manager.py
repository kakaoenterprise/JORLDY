from manager.eval_manager import EvalManager


def test_eval_manager(MockEnv, env_config, MockAgent, agent_config):
    # test init
    eval_iteration = 5
    record, record_period = True, 10
    eval_manager = EvalManager(
        Env=MockEnv,
        env_config=env_config,
        iteration=eval_iteration,
        record=record,
        record_period=record_period,
    )

    # test after init
    assert eval_iteration == eval_manager.iteration
    assert eval_manager.record == True

    # test evaluate
    agent = MockAgent(**agent_config)
    steps = [10, 20, 29]
    for step in steps:
        score, frames = eval_manager.evaluate(agent=agent, step=step)

        assert isinstance(score, float)
        if step == 29:
            assert len(frames) == 0
        else:
            assert len(frames) == env_config["episode_len"]

        frames.clear()
