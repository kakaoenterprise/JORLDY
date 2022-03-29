### RND PPO Super Mario Bros Config ###

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
    "name": "rnd_ppo",
    "network": "discrete_policy_separate_value",
    "head": "cnn",
    "gamma": 0.99,
    "batch_size": 32,
    "n_step": 128,
    "n_epoch": 3,
    "_lambda": 0.95,
    "epsilon_clip": 0.1,
    "vf_coef": 1.0,
    "ent_coef": 0.001,
    "clip_grad_norm": 1.0,
    "use_standardization": False,
    "lr_decay": True,
    # Parameters for Random Network Distillation
    "rnd_network": "rnd_cnn",  # rnd_mlp, rnd_cnn, rnd_multi
    "gamma_i": 0.99,
    "extrinsic_coeff": 1.0,
    "intrinsic_coeff": 1.0,
    "obs_normalize": True,
    "ri_normalize": True,
    "batch_norm": True,
    "non_episodic": True,
    "non_extrinsic": False,
}

optim = {
    "name": "adam",
    "lr": 0.0001,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 30000000,
    "print_period": 10000,
    "save_period": 100000,
    "eval_iteration": 1,
    "record": True,
    "record_period": 500000,
    # distributed setting
    "distributed_batch_size": 256,
    "update_period": agent["n_step"],
    "num_workers": 64,
}
