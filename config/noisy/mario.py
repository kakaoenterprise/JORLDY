### Noisy DQN Mario Config ###

env = {
    "name": "mario",
    "render": False,
    "gray_img": True,
    "img_width": 84,
    "img_height": 84,
    "stack_frame": 4,
}

agent = {
    "name": "noisy",
    "network": "noisy_cnn",
    "optimizer": "adam",
    "learning_rate": 5e-4,
    "gamma": 0.99,
    "explore_step": 1000000,
    "buffer_size": 100000,
    "batch_size": 32,
    "start_train_step": 50000,
    "target_update_period": 10000,
}

train = {
    "training" : True,
    "load_path" : None,
    "run_step" : 100000000,
    "print_period" : 10000,
    "save_period" : 50000,
    "test_iteration": 3,
    "record" : True,
    # distributed setting
    "update_period" : 32,
    "num_worker" : 16,
}
