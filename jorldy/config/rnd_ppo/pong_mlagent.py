### RND PPO Pong_ML-Agents Config ###

env = {"name": "pong_mlagent", "time_scale": 12.0}

agent = {
    "name": "rnd_ppo",
    "network": "discrete_policy_separate_value",
    "gamma": 0.99,
    "batch_size": 64,
    "n_step": 128,
    "n_epoch": 3,
    "_lambda": 0.95,
    "epsilon_clip": 0.1,
    "vf_coef": 0.5,
    "ent_coef": 0.0,
    "clip_grad_norm": 1.0,
    "lr_decay": True,
    # Parameters for Random Network Distillation
    "rnd_network": "rnd_mlp",  # rnd_mlp, rnd_cnn, rnd_multi
    "gamma_i": 0.99,
    "extrinsic_coeff": 2.0,
    "intrinsic_coeff": 1.0,
    "obs_normalize": True,
    "ri_normalize": True,
    "batch_norm": True,
}

optim = {
    "name": "adam",
    "lr": 0.0001,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 200000,
    "print_period": 5000,
    "save_period": 50000,
    "eval_iteration": 10,
    # distributed setting
    "distributed_batch_size": 512,
    "update_period": agent["n_step"],
    "num_workers": 8,
}
