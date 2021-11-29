### Noisy DQN CartPole Config ###

env = {
    "name": "cartpole",
    "action_type": "discrete",
    "render": False,
}

agent = {
    "name": "noisy",
    "network": "noisy",
    "gamma": 0.99,
    "explore_ratio": 0.1,
    "buffer_size": 50000,
    "batch_size": 32,
    "start_train_step": 2000,
    "target_update_period": 500,
    # noisy
    "noise_type": "factorized",  # [independent, factorized]
}

optim = {
    "name": "adam",
    "lr": 0.0001,
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
