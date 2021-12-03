import shutil
import numpy as np

from manager.log_manager import LogManager


def test_log_manager():
    env_name, id, experiment = "mock_env", "mock_agent", "tmp_test"
    log_manager = LogManager(env=env_name, id=id, experiment=experiment)

    # test after init
    assert env_name in log_manager.path
    assert id in log_manager.path
    assert experiment in log_manager.path

    # test write
    scalar_dict = {
        "mock_metric1": np.random.random(),
        "mock_metric2": np.random.random(),
        "score": np.random.random(),
    }
    frames = np.random.randint(low=0, high=255, size=(60, 32, 32, 3))
    log_manager.write(scalar_dict=scalar_dict, frames=frames, step=100)

    shutil.rmtree(f"./logs/{experiment}")
