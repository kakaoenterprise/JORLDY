### DQN BreakOut Config ###

env = {
    "name": "breakout",
    "render": False,
    "gray_img": True,
    "img_width": 84,
    "img_height": 84,
    "stack_frame": 4,
}

agent = {
    "name": "dqn",
    "network": "dqn_cnn",
    "optimizer": "adam",
    "learning_rate": 1e-4, #0.00025,
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.01,
    "explore_step": 1000000,
    "buffer_size": 1000000,
    "batch_size": 32,
    "start_train_step": 100000,
    "target_update_term": 10000,
}

train = {
    "training" : True,
    "load_path" : None, #"./logs/breakout/dqn/tmp/",
    "run_step" : 100000000,
    "print_term" : 10000,
    "save_term" : 100000,
    "update_term" : 10,
    "test_iteration": 5,
    "num_worker" : 16,
}