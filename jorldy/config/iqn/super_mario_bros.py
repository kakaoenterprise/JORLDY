### IQN Super Mario Bros Config ###

env = {
    "name": "super_mario_bros",
    "render": False,
    "gray_img": True,
    "img_width": 84,
    "img_height": 84,
    "stack_frame": 4,
}

agent = {
    "name": "iqn",
    "network": "iqn",
    "head": "cnn",
    "optimizer": "adam",
    "opt_eps": 1e-2 / 32,
    "learning_rate": 0.00005,
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.01,
    "explore_ratio": 0.1,
    "buffer_size": 1000000,
    "batch_size": 32,
    "start_train_step": 100000,
    "target_update_period": 10000,
    "num_sample": 64,
    "embedding_dim": 64,
    "sample_min": 0.0,
    "sample_max": 1.0,
    "lr_decay": True,
}

optim = {
    "name": "adam",
    "eps": 1e-2 / agent["batch_size"],
    "lr": 5e-5,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 100000000,
    "print_period": 5000,
    "save_period": 50000,
    "eval_iteration": 5,
    # distributed setting
    "update_period": 8,
    "num_workers": 16,
}
