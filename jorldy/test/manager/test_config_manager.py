from manager.config_manager import ConfigManager
import mock_config


def test_config_manager():
    mock_unknown = [
        "--train.num_workers=4",  # change var using =
        "--agent.batch_size",  # set new var using space
        "16",
        "--env.render",  # change var using space
        "True",
        "--optim.mock_var=1e-4",  # set new var using =
    ]

    manager = ConfigManager("mock_config", unknown_args=mock_unknown)
    config = manager.config

    # test parsing
    for key, val in config.env.items():
        if key == "render":
            assert key in mock_config.env.keys()
            assert val == True
        else:
            assert key in mock_config.env.keys()
            assert val == mock_config.env[key]

    for key, val in config.agent.items():
        if key == "batch_size":
            assert key not in mock_config.agent.keys()
            assert val == 16
        else:
            assert key in mock_config.agent.keys()
            assert val == mock_config.agent[key]

    for key, val in config.optim.items():
        if key == "mock_var":
            assert key not in mock_config.optim.keys()
            assert val == 1e-4
        else:
            assert key in mock_config.optim.keys()
            assert val == mock_config.optim[key]

    for key, val in config.train.items():
        if key == "num_workers":
            assert key in mock_config.train.keys()
            assert val == 4
        else:
            assert key in mock_config.train.keys()
            assert val == mock_config.train[key]
