### ICM PPO Pong_ML-Agents Config ###

env = {"name": "pong_mlagent", "train_mode": True}

agent = {
    "name": "icm_ppo",
    "network": "discrete_policy_value",
    "gamma": 0.99,
    "batch_size": 32,
    "n_step": 128,
    "n_epoch": 3,
    "_lambda": 0.95,
    "epsilon_clip": 0.1,
    "vf_coef": 1.0,
    "ent_coef": 0.1,
    "clip_grad_norm": 1.0,
    # Parameters for Curiosity-driven Exploration
    "icm_network": "icm_mlp",  # icm_mlp, icm_cnn, icm_multi
    "beta": 0.2,
    "lamb": 1.0,
    "eta": 0.1,
    "extrinsic_coeff": 1.0,
    "intrinsic_coeff": 1.0,
}

optim = {
    "name": "adam",
    "lr": 2.5e-4,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 100000,
    "print_period": 1000,
    "save_period": 10000,
    "eval_iteration": 10,
    # distributed setting
    "distributed_batch_size": 256,
    "update_period": agent["n_step"],
    "num_workers": 8,
}
