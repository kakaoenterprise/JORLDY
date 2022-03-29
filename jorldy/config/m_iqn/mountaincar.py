### Munchausen IQN MountainCar Config ###

env = {
    "name": "mountain_car",
    "render": False,
}

agent = {
    "name": "m_iqn",
    "network": "iqn",
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.1,
    "explore_ratio": 0.1,
    "buffer_size": 10000,
    "batch_size": 64,
    "start_train_step": 10000,
    "target_update_period": 500,
    "num_sample": 64,
    "embedding_dim": 64,
    "sample_min": 0.0,
    "sample_max": 1.0,
    "lr_decay": True,
    # M-DQN Parameters
    "alpha": 0.9,
    "tau": 0.03,
    "l_0": -1,
}

optim = {
    "name": "adam",
    "eps": 1e-2 / agent["batch_size"],
    "lr": 5e-5,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 100000,
    "print_period": 1000,
    "save_period": 10000,
    "eval_iteration": 5,
    # distributed setting
    "update_period": 32,
    "num_workers": 8,
}
