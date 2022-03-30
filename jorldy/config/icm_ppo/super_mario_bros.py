### ICM PPO Super Mario Bros Config ###

env = {
    "name": "super_mario_bros",
    "render": False,
    "gray_img": True,
    "img_width": 84,
    "img_height": 84,
    "stack_frame": 4,
    "no_op": True,
    "skip_frame": 4,
    "reward_clip": True,
    "episodic_life": True,
}

agent = {
    "name": "icm_ppo",
    "network": "discrete_policy_value",
    "head": "cnn",
    "gamma": 0.99,
    "batch_size": 16,
    "n_step": 128,
    "n_epoch": 3,
    "_lambda": 0.95,
    "epsilon_clip": 0.1,
    "vf_coef": 1.0,
    "ent_coef": 0.01,
    "clip_grad_norm": 1.0,
    "use_standardization": True,
    "lr_decay": True,
    # Parameters for Curiosity-driven Exploration
    "icm_network": "icm_cnn",  # icm_mlp, icm_cnn, icm_multi
    "beta": 0.2,
    "lamb": 1.0,
    "eta": 0.1,
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
    "save_period": 500000,
    "eval_iteration": 1,
    "record": True,
    "record_period": 500000,
    # distributed setting
    "distributed_batch_size": 1024,
    "update_period": agent["n_step"],
    "num_workers": 64,
}
