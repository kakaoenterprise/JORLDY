### PER Atari Config ###

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
    "name": "per",
    "network": "discrete_q_network",
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
    "lr_decay": True,
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
    "eval_time_limit": None,
    "record": True,
    "record_period": 300000,
    # distributed setting
    "update_period": agent["learn_period"],
    "num_workers": 16,
}
