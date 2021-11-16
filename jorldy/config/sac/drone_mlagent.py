### SAC Drone Config ###

env = {"name": "drone_mlagent", "train_mode": True}

agent = {
    "name": "sac",
    "actor": "continuous_policy",
    "critic": "sac_critic",
    "head": "multi",
    "use_dynamic_alpha": True,
    "gamma": 0.99,
    "tau": 5e-3,
    "buffer_size": 50000,
    "batch_size": 64,
    "start_train_step": 25000,
    "static_log_alpha": -2.0,
}

optim = {
    "actor": "adam",
    "critic": "adam",
    "alpha": "adam",
    "actor_lr": 5e-4,
    "critic_lr": 1e-3,
    "alpha_lr": 3e-4,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 30000000,
    "print_period": 10000,
    "save_period": 100000,
    "eval_iteration": 5,
    "record": False,
    "record_period": 300000,
    # distributed setting
    "update_period": 128,
    "num_workers": 16,
}
