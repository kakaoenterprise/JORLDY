### SAC Atari Config ###

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
    "name": "sac",
    "actor": "discrete_policy",
    "critic": "discrete_q_network",
    "head": "cnn",
    "use_dynamic_alpha": True,
    "gamma": 0.99,
    "tau": 5e-3,
    "buffer_size": 1000000,
    "batch_size": 32,
    "start_train_step": 100000,
    "static_log_alpha": -6.0,
    "target_update_period": 10000,
}


optim = {
    "actor": "adam",
    "critic": "adam",
    "alpha": "adam",
    # "actor_lr": 5e-4,
    # "critic_lr": 1e-3,
    # "alpha_lr": 3e-4,
    "actor_lr": 1.5e-4,
    "critic_lr": 3e-4,
    "alpha_lr": 1e-5,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 10000000,
    "print_period": 10000,
    "save_period": 100000,
    "eval_iteration": 5,
    "eval_time_limit": None,
    "record": True,
    "record_period": 500000,
    # distributed setting
    "update_period": 32,
    "num_workers": 16,
}
