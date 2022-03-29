### Ape-X MountainCar Config ###

env = {
    "name": "mountain_car",
    "render": False,
}

agent = {
    "name": "ape_x",
    "network": "dueling",
    "gamma": 0.99,
    "buffer_size": 50000,
    "batch_size": 32,
    "clip_grad_norm": 40.0,
    "start_train_step": 2000,
    "target_update_period": 1000,
    "lr_decay": True,
    # MultiStep
    "n_step": 3,
    # PER
    "alpha": 0.6,
    "beta": 0.4,
    "uniform_sample_prob": 1e-3,
}

optim = {
    "name": "rmsprop",
    "eps": 1.5e-7,
    "lr": 2.5e-4 / 4,
    "centered": True,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 100000,
    "print_period": 1000,
    "save_period": 10000,
    "eval_iteration": 10,
    # distributed setting
    "distributed_batch_size": 512,
    "update_period": 16,
    "num_workers": 32,
}
