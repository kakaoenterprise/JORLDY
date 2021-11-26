### PER Super Mario Bros Config ###

env = {
    "name": "super_mario_bros",
    "render": False,
    "gray_img": True,
    "img_width": 84,
    "img_height": 84,
    "stack_frame": 4,
}

agent = {
    "name": "per",
    "network": "dqn",
    "head": "cnn",
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.01,
    "explore_ratio": 0.1,
    "buffer_size": 1000000,
    "batch_size": 32,
    "start_train_step": 100000,
    "target_update_period": 10000,
    "alpha": 0.6,
    "beta": 0.4,
    "learn_period": 16,
    "uniform_sample_prob": 1e-3,
}

optim = {
    "name": "adam",
    "lr": 2.5e-4 / 4,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 30000000,
    "print_period": 10000,
    "save_period": 100000,
    "eval_iteration": 5,
    "record": True,
    "record_period": 300000,
    # distributed setting
    "update_period": agent["learn_period"],
    "num_workers": 16,
}
