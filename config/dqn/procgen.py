### DQN Procgen Config ###

env = {
    # "name": it should be defined in the command. ex) python main.py --config config.dqn.procgen --env.name coinrun
    "render": False,
    "gray_img": True,
    "stack_frame": 4,
    "no_op": False,
    "reward_clip": True,
}

agent = {
    "name": "dqn",
    "network": "dqn",
    "head": "cnn",
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.1,
    "explore_step": 1000000,
    "buffer_size": 1000000,
    "batch_size": 32,
    "start_train_step": 100000,
    "target_update_period": 10000,
}

optim = {
    "name": "rmsprop",
    "lr": 2.5e-4,
}

train = {
    "training" : True,
    "load_path" : None,
    "run_step" : 30000000,
    "print_period" : 10000,
    "save_period" : 100000,
    "test_iteration": 5,
    "record" : True,
    "record_period" : 300000,
    # distributed setting
    "update_period" : 32,
    "num_workers" : 16,
}