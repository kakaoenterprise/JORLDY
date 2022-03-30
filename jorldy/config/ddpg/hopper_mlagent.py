### DDPG Hopper Config ###

env = {"name": "hopper_mlagent", "time_scale": 12.0}

agent = {
    "name": "ddpg",
    "actor": "deterministic_policy",
    "critic": "continuous_q_network",
    "gamma": 0.99,
    "buffer_size": 50000,
    "batch_size": 128,
    "start_train_step": 2000,
    "tau": 1e-3,
    "lr_decay": True,
    # OU noise
    "mu": 0,
    "theta": 1e-3,
    "sigma": 2e-3,
}

optim = {
    "actor": "adam",
    "critic": "adam",
    "actor_lr": 5e-4,
    "critic_lr": 1e-3,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 300000,
    "print_period": 5000,
    "save_period": 10000,
    "eval_iteration": 10,
    # distributed setting
    "distributed_batch_size": 256,
    "update_period": 1,
    "num_workers": 8,
}
