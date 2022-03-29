### V-MPO Mujoco Config ###

env = {
    # "name": it should be defined in the command. ex) python main.py --config config.AGENT.mujoco --env.name half_cheetah
    "render": False,
}

agent = {
    "name": "vmpo",
    "network": "continuous_policy_value",
    "gamma": 0.99,
    "batch_size": 64,
    "n_step": 128,
    "n_epoch": 1,
    "_lambda": 0.95,
    "min_eta": 1e-8,
    "min_alpha_mu": 1e-8,
    "min_alpha_sigma": 1e-8,
    "eps_eta": 0.01,
    "eps_alpha_mu": 0.01,
    "eps_alpha_sigma": 5 * 1e-5,
    "eta": 1.0,
    "alpha_mu": 1.0,
    "alpha_sigma": 1.0,
    "lr_decay": True,
}

optim = {
    "name": "adam",
    "lr": 5e-4,
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
