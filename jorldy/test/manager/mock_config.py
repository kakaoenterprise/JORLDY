### Mock Config ###

env = {
    "name": "mock_env",
    "render": False,
}

agent = {
    "name": "mock_agent",
    "network": "mock_network",
}

optim = {
    "name": "mock_optim",
    "lr": 0.0001,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 100000,
    "print_period": 1000,
    "save_period": 10000,
    "eval_iteration": 10,
    "record": False,
    "record_period": None,
    # distributed setting
    "update_period": 32,
    "num_workers": 8,
}
