### R2D2 Atari Config ###

env = {
    # "name": it should be defined in the command. ex) python main.py --config config.AGENT.atari --env.name breakout
    "render": False,
    "gray_img": True,
    "img_width": 84,
    "img_height": 84,
    "stack_frame": 4,
    "no_op": False,
    "skip_frame": 4,
    "reward_clip": True,
    "episodic_life": True,
}

agent = {
    "name": "r2d2",
    "network": "r2d2",
    "head": "cnn",
    "gamma": 0.99,
    "buffer_size": 2000000,
    "batch_size": 32,
    "clip_grad_norm": 40.0,
    "start_train_step": 100000,
    "target_update_period": 10000,
    # MultiStep
    "n_step": 3,
    # PER
    "alpha": 0.9,
    "beta": 0.6,
    "uniform_sample_prob": 1e-3,
    # R2D2
    "seq_len": 6,
    "n_burn_in": 3,
    "zero_padding": True,
}

optim = {
    "name": "adam",
    "lr": 1e-4,
}

train = {
    "training" : True,
    "load_path" : None,
    "run_step" : 30000000,
    "print_period" : 10000,
    "save_period" : 100000,
    "eval_iteration": 5,
    "record" : True,
    "record_period" : 300000,
    # distributed setting
    "distributed_batch_size" : 512,
    "update_period" : 16,
    "num_workers" : 8,
}