### ICM PPO Mujoco Config ###

env = {
    # "name": it should be defined in the command. ex) python main.py --config config.AGENT.mujoco --env.name half_cheetah
    "render": False,
}

agent = {
    "name": "icm_ppo",
    "network": "continuous_policy_separate_value",
    "head": "mlp",
    "gamma": 0.99,
    "batch_size": 32,
    "n_step": 128,
    "n_epoch": 3,
    "_lambda": 0.95,
    "epsilon_clip": 0.1,
    "vf_coef": 1.0,
    "ent_coef": 0.01,
    "clip_grad_norm": 1.0,
    "lr_decay": True,
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
    "run_step": 30000000,
    "print_period": 10000,
    "save_period": 100000,
    "eval_iteration": 5,
    "record": True,
    "record_period": 300000,
    # distributed setting
    "distributed_batch_size": 1024,
    "update_period": agent["n_step"],
    "num_workers": 32,
}
