### Noisy DQN Assualt Config ###

env = {
    "name": "assault",
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
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "explore_step": 1000000,
    "buffer_size": 1000000,
    "batch_size": 32,
    "start_train_step": 100000,
    "target_update_term": 10000
}

train = {
    "training" : True,
    "load_path" : None,
    "train_step" : 100000000,
    "test_step" : 1000000,
    "print_term" : 50,
    "save_term" : 500,
    "test_iteration": 5,
}