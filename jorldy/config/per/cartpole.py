### PER CartPole Config ###

env = {
    "name": "cartpole",
    "action_type": "discrete",
    "render": False,
}

agent = {
    "name": "per",
    "network": "discrete_q_network",
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.01,
    "explore_ratio": 0.2,
    "buffer_size": 50000,
    "batch_size": 32,
    "start_train_step": 2000,
    "target_update_period": 500,
    "alpha": 0.6,
    "beta": 0.4,
    "learn_period": 2,
    "uniform_sample_prob": 1e-3,
    "lr_decay": True,
}

optim = {
    "name": "adam",
    "lr": 0.0001,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 100000,
    "print_period": 1000,
    "save_period": 10000,
    "eval_iteration": 5,
    # distributed setting
    "update_period": agent["learn_period"],
    "num_workers": 8,
}
