### QRDQN Pong Config ###

env = {
    "name": "pong",
    "render": False,
    "gray_img": True,
    "img_width": 80,
    "img_height": 80,
    "stack_frame": 4,
}

agent = {
    "name": "qrdqn",
    "network": "dqn_cnn",
    "optimizer": "adam",
    "opt_eps": 1e-2/32,
    "learning_rate": 0.00005,
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.01,
    "explore_step": 1000000,
    "buffer_size": 1000000,
    "batch_size": 32,
    "start_train_step": 100000,
    "target_update_term": 10000,
    "num_support": 200
}

train = {
    "training" : True,
    "load_path" : None, #"./logs/breakout/dqn/20201027142347/",
    "train_step" : 10000000,
    "test_step" : 1000000,
    "print_term" : 10,
    "save_term" : 100,
    "test_iteration": 10,
}