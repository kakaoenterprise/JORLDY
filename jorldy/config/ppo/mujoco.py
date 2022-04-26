### PPO Mujoco Config ###

env = {
    # "name": it should be defined in the command. ex) python main.py --config config.AGENT.mujoco --env.name half_cheetah
    "render": False,
}

agent = {
    "name": "ppo",
    "network": "continuous_policy_value",
    "gamma": 0.99,
    "batch_size": 512,
    "n_step": 2048,
    "n_epoch": 10,
    "_lambda": 0.95,
    "epsilon_clip": 0.1,
    "vf_coef": 1.0,
    "ent_coef": 0.01,
    "clip_grad_norm": 1.0,
    "lr_decay": True,
}

optim = {
    "name": "adam",
    "lr": 3e-4,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 1000000,
    "print_period": 10000,
    "save_period": 100000,
    "eval_iteration": 10,
    "record": True,
    "record_period": 500000,
    # distributed setting
    "distributed_batch_size": 2048,
    "update_period": agent["n_step"],
    "num_workers": 32,
}
