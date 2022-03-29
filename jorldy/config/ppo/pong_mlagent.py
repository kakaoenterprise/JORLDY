### PPO Pong_ML-Agents Config ###

env = {"name": "pong_mlagent", "time_scale": 12.0}

agent = {
    "name": "ppo",
    "network": "discrete_policy_value",
    "gamma": 0.99,
    "batch_size": 32,
    "n_step": 128,
    "n_epoch": 3,
    "_lambda": 0.95,
    "epsilon_clip": 0.1,
    "vf_coef": 1.0,
    "ent_coef": 0.01,
    "lr_decay": True,
}

optim = {
    "name": "adam",
    "lr": 2.5e-4,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 200000,
    "print_period": 1000,
    "save_period": 50000,
    "eval_iteration": 5,
    # distributed setting
    "distributed_batch_size": 256,
    "update_period": agent["n_step"],
    "num_workers": 8,
}
