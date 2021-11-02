### Double DQN MountainCar Config ###

env = {
    "name":"mountain_car",
    "render":False,
}

agent = {
    "name": "double",
    "network": "dqn",
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.1,
    "explore_step": 20000,
    "buffer_size": 10000,
    "batch_size": 32,
    "start_train_step": 10000,
    "target_update_period": 200,
}

optim = {
    "name": "adam",
    "lr": 5e-4,
}

train = {
    "training" : True,
    "load_path" : None,
    "run_step" : 100000,
    "print_period" : 1000,
    "save_period" : 10000,
    "eval_iteration": 5,
    # distributed setting
    "update_period" : 32,
    "num_workers" : 8,
}
