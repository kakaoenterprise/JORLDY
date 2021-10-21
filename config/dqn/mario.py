### DQN Mario Config ###

env = {
    "name": "mario",
    "render": False,
    "gray_img": True,
    "img_width": 84,
    "img_height": 84,
    "stack_frame": 4,
}

agent = {
    "name": "dqn",
    "network": "dqn",
    "head": "cnn",
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.01,
    "explore_step": 1000000,
    "buffer_size": 1000000,
    "batch_size": 32,
    "start_train_step": 100000,
    "target_update_period": 10000,
}

optim = {
    "name": "adam",
    "lr": 1e-4,
}

train = {
    "training" : True,
    "load_path" : None,
    "run_step" : 100000000,
    "print_period" : 5000,
    "save_period" : 50000,
    "eval_iteration": 1,
    "record" : True,
    "record_period" : 200000,
    # distributed setting
    "update_period" : 32,
    "num_workers" : 16,
}
