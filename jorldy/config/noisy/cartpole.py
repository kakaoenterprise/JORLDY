### Noisy DQN CartPole Config ###

env = {
    "name": "cartpole",
    "mode": "discrete",
    "render": False,
}

agent = {
    "name": "noisy",
    "network": "noisy",
    "gamma": 0.99,
    "explore_step": 20000,
    "buffer_size": 10000,
    "batch_size": 32,
    "start_train_step": 10000,
    "target_update_period": 200,
    # noisy
    "noise_type": "independent",  # [independent, factorized]
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
