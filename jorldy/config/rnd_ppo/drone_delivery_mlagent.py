### RND PPO Drone Delivery Config ###

env = {"name": "drone_delivery_mlagent", "train_mode": True}

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
    # Parameters for Random Network Distillation
    "rnd_network": "rnd_multi",  # rnd_mlp, rnd_cnn, rnd_multi
    "gamma_i": 0.99,
    "extrinsic_coeff": 1.0,
    "intrinsic_coeff": 1.0,
    "obs_normalize": True,
    "ri_normalize": True,
    "batch_norm": True,
}

optim = {
    "name": "adam",
    "lr": 2.5e-4,
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
    "distributed_batch_size": 256,
    "update_period": agent["n_step"],
    "num_workers": 4,
}
