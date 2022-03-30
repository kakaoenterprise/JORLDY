### RND PPO Mujoco Config ###

env = {
    # "name": it should be defined in the command. ex) python main.py --config config.AGENT.mujoco --env.name half_cheetah
    "render": False,
}

agent = {
    "name": "rnd_ppo",
    "network": "continuous_policy_separate_value",
    "head": "mlp",
    "gamma": 0.999,
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
    "run_step": 100000000,
    "print_period": 10000,
    "save_period": 100000,
    "eval_iteration": 1,
    "record": True,
    "record_period": 1000000,
    # distributed setting
    "distributed_batch_size": 1024,
    "update_period": agent["n_step"],
    "num_workers": 64,
}
