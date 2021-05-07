### PER Seaquest Config ###

env = {
    "name": "seaquest",
    "render": False,
    "gray_img": True,
    "img_width": 80,
    "img_height": 80,
    "stack_frame": 4,
}

agent = {
    "name": "per",
    "network": "dqn_cnn",
    "optimizer": "adam",
    "learning_rate": 0.00025/4,
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.01,
    "explore_step": 1000000,
    "buffer_size": 1000000,
    "batch_size": 32,
    "start_train_step": 100000,
    "target_update_period": 10000,
    "alpha": 0.6,
    "beta": 0.4,
    "learn_period": 4,
    "uniform_sample_prob": 1e-3,
}

train = {
    "training" : True,
    "load_path" : None, #"./logs/breakout/dqn/20201027142347/",
    "train_step" : 10000000,
    "test_step" : 1000000,
    "print_period" : 10,
    "save_period" : 100,
    "test_iteration": 10,
}