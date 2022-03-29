### R2D2 CartPole Config ###

env = {
    "name": "cartpole",
    "action_type": "discrete",
    "render": False,
}

agent = {
    "name": "r2d2",
    "network": "r2d2",
    "head": "mlp",
    "gamma": 0.99,
    "buffer_size": 50000,
    "batch_size": 64,
    "clip_grad_norm": 40.0,
    "start_train_step": 2000,
    "target_update_period": 500,
    "lr_decay": True,
    # MultiStep
    "n_step": 3,
    # PER
    "alpha": 0.6,
    "beta": 0.6,
    "uniform_sample_prob": 1e-3,
    # R2D2
    "seq_len": 4,
    "n_burn_in": 1,
    "zero_padding": True,
}

optim = {
    "name": "adam",
    "lr": 0.0001,
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
    "num_workers": 16,
}
