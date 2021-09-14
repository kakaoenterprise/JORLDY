### R2D2 Procgen Config ###

env = {
    # "name": it should be defined in the command. ex) python main.py --config config.AGENT.procgen --env.name coinrun
    "render": False,
    "gray_img": True,
    "stack_frame": 4,
    "no_op": False,
    "reward_clip": True,
}

agent = {
    "name": "r2d2",
    "network": "r2d2",
    "head": "cnn",
    "gamma": 0.997,
    "buffer_size": 2000000,
    "batch_size": 64,
    "clip_grad_norm": 40.0,
    "start_train_step": 50000,
    "target_update_period": 2500,
    # MultiStep
    "n_step": 5,
    # PER
    "alpha": 0.6,
    "beta": 0.4,
    "uniform_sample_prob": 1e-3,
    # Sequence Length
    "m": 80,
}

optim = {
    "name": "adam",
    "eps": 1e-4,
    "lr": 1e-4,
}

train = {
    "training" : True,
    "load_path" : None,
    "run_step" : 30000000,
    "print_period" : 10000,
    "save_period" : 100000,
    "test_iteration": 5,
    "record" : True,
    "record_period" : 300000,
    # distributed setting
    "distributed_batch_size" : 512,
    "update_period" : 100,
    "num_workers" : 128,
}