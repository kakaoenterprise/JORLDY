### DQN Seaquest Config ###

env = {
    "name": "seaquest",
    "render": False,
    "gray_img": True,
    "img_width": 80,
    "img_height": 80,
    "stack_frame": 4,
}

agent = {
    "name": "dqn",
    "network": "dqn_cnn",
    "optimizer": "adam",
    "learning_rate": 5e-4,
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.1,
    "explore_step": 1000000,
    "buffer_size": 100000,
    "batch_size": 64,
    "start_train_step": 100000,
    "target_update_term": 10000,
}

train = {
    "training" : True,
    "load_path" : None,
    "train_step" : 10000000,
    "test_step" : 1000000,
    "print_term" : 10,
    "save_term" : 100,
    "test_iteration": 10,
}