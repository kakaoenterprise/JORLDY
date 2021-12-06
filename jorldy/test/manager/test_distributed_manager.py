from manager.distributed_manager import DistributedManager


def test_distributed_manager(MockEnv, env_config, MockAgent, agent_config):
    # test init
    num_workers, mode = 2, "sync"
    distributed_manager = DistributedManager(
        Env=MockEnv,
        env_config=env_config,
        Agent=MockAgent,
        agent_config=agent_config,
        num_workers=num_workers,
        mode=mode,
    )

    # test after init
    assert len(distributed_manager.actors) == num_workers

    # can not test run
    distributed_manager.terminate()
