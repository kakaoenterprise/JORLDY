### SAC Pong_ML-Agents Config ###

env = {"name": "pong_mlagent", "time_scale": 12.0}

agent = {
    "name": "sac",
    "actor": "discrete_policy",
    "critic": "discrete_q_network",
    "use_dynamic_alpha": True,
    "gamma": 0.99,
    "tau": 5e-3,
    "buffer_size": 50000,
    "batch_size": 32,
    "start_train_step": 25000,
    "static_log_alpha": -3.0,
    "target_update_period": 1000,
}

optim = {
    "actor": "adam",
    "critic": "adam",
    "alpha": "adam",
    "actor_lr": 1.5e-4,
    "critic_lr": 3e-4,
    "alpha_lr": 1e-5,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 200000,
    "print_period": 5000,
    "save_period": 50000,
    "eval_iteration": 10,
    # distributed setting
    "update_period": 8,
    "num_workers": 16,
}
