from manager.distributed_manager import DistributedManager
import time


def test_sync_distributed_manager(MockEnv, env_config, MockAgent, agent_config):
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
    time.sleep(1)


def test_async_distributed_manager(MockEnv, env_config, MockAgent, agent_config):
    # test init
    num_workers, mode = 2, "async"
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
    time.sleep(1)
