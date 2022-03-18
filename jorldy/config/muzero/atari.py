### MuZero Atari Config ###

env = {
    # "name": it should be defined in the command. ex) python main.py --config config.AGENT.atari --env.name breakout
    "render": False,
    "gray_img": True,
    "img_width": 96,
    "img_height": 96,
    "stack_frame": 1,
    "no_op": True,
    "skip_frame": 4,
    "reward_clip": True,
    "episodic_life": True,
}

agent = {
    "name": "mu_zero",
    "network": "mu_zero_resnet",
    "head": "downsample",
    "gamma": 0.997,
    "epsilon_init": 1.0,
    "epsilon_min": 0.1,
    "explore_ratio": 0.1,
    "buffer_size": 10000,
    "batch_size": 8,
    "start_train_step": 0,
    "target_update_period": 10000,
}

optim = {
    "name": "adam",
    "lr": 1e-4,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 10000000,
    "print_period": 100,
    "save_period": 100000,
    "eval_iteration": 5,
    "record": True,
    "record_period": 500000,
    # distributed setting
    "update_period": 200,
    "num_workers": 8,
}
