### REINFORCE CartPole Config ###

env = {
    "name": "cartpole",
    "action_type": "discrete",
    "render": False,
}

agent = {
    "name": "reinforce",
    "network": "discrete_policy",
    "gamma": 0.99,
    "lr_decay": True,
}

optim = {
    "name": "adam",
    "lr": 1e-4,
}


train = {
    "training": True,
    "load_path": None,
    "run_step": 100000,
    "print_period": 1000,
    "save_period": 10000,
    "eval_iteration": 10,
}
