### PER CartPole Config ###

env = {
    "name":"cartpole",
    "mode":"discrete",
    "render":False,
}

agent = {
    "name": "per",
    "network": "dqn",
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.01,
    "explore_step": 20000,
    "buffer_size": 50000,
    "batch_size": 32,
    "start_train_step": 2000,
    "target_update_period": 1000,
    "alpha": 0.6,
    "beta": 0.4,
    "learn_period": 16,
    "uniform_sample_prob": 1e-3,
}

optim = {
    "name": "adam",
    "lr": 2.5e-4/4,
}

train = {
    "training" : True,
    "load_path" : None,
    "run_step" : 100000,
    "print_period" : 1000,
    "save_period" : 10000,
    "eval_iteration": 5,
    # distributed setting
    "update_period" : agent["learn_period"],
    "num_workers" : 8,
}
