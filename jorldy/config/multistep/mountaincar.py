### Multistep DQN MountainCar Config ###

env = {
    "name": "mountain_car",
    "render": False,
}

agent = {
    "name": "multistep",
    "network": "discrete_q_network",
    "n_step": 4,
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.1,
    "explore_ratio": 0.1,
    "buffer_size": 10000,
    "batch_size": 32,
    "start_train_step": 10000,
    "target_update_period": 200,
    "lr_decay": True,
}

optim = {
    "name": "adam",
    "lr": 2.5e-4,
}
train = {
    "training": True,
    "load_path": None,
    "run_step": 100000,
    "print_period": 1000,
    "save_period": 10000,
    "eval_iteration": 5,
    # distributed setting
    "update_period": 8,
    "num_workers": 8,
}
