### Rainbow IQN Atari Config ###

env = {
    # "name": it should be defined in the command. ex) python main.py --config config.AGENT.atari --env.name breakout
    "render": False,
    "gray_img": True,
    "img_width": 84,
    "img_height": 84,
    "stack_frame": 4,
    "no_op": True,
    "skip_frame": 4,
    "reward_clip": True,
    "episodic_life": True,
}

agent = {
    "name": "rainbow_iqn",
    "network": "rainbow_iqn",
    "head": "cnn",
    "gamma": 0.99,
    "buffer_size": 1000000,
    "batch_size": 32,
    "start_train_step": 100000,
    "target_update_period": 10000,
    "lr_decay": True,
    # MultiStep
    "n_step": 3,
    # PER
    "alpha": 0.6,
    "beta": 0.4,
    "learn_period": 4,
    "uniform_sample_prob": 1e-3,
    # Noisy
    "noise_type": "factorized",  # [independent, factorized]
    # IQN
    "num_sample": 64,
    "embedding_dim": 64,
    "sample_min": 0.0,
    "sample_max": 1.0,
}

optim = {
    "name": "adam",
    "eps": 1e-2 / agent["batch_size"],
    "lr": 2.5e-4 / 4,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 30000000,
    "print_period": 10000,
    "save_period": 100000,
    "eval_iteration": 5,
    "eval_time_limit": None,
    "record": True,
    "record_period": 300000,
    # distributed setting
    "update_period": 32,
    "num_workers": 16,
}
