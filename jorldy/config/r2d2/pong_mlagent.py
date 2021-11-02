### R2D2 Pong_ML-Agents Config ###

env = {
    "name": "pong_mlagent",
    "train_mode": True
}

agent = {
    "name": "r2d2",
    "network": "r2d2",
    "head": "mlp",
    "gamma": 0.997,
    "buffer_size": 2000000,
    "batch_size": 64,
    "clip_grad_norm": 40.0,
    "start_train_step": 50000,
    "target_update_period": 2500,
    # MultiStep
    "n_step": 5,
    # PER
    "alpha": 0.9,
    "beta": 0.6,
    "uniform_sample_prob": 1e-3,
    # R2D2
    "seq_len": 7,
    "n_burn_in": 4,
}

optim = {
    "name": "adam",
    "eps": 1e-3,
    "lr": 1e-4,
}

train = {
    "training" : True,
    "load_path" : None,
    "run_step" : 200000,
    "print_period" : 5000,
    "save_period" : 50000,
    "eval_iteration": 10,
    # distributed setting
    "distributed_batch_size" : 512,
    "update_period" : 16,
    "num_workers" : 8,
}