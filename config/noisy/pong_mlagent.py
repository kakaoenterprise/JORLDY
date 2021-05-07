### Noisy DQN Pong_ML-Agents Config ###

env = {
    "name": "pong_mlagent",
    "train_mode": True
}

agent = {
    "name": "noisy",
    "network": "noisy",
    "optimizer": "adam",
    "learning_rate": 0.00025,
    "gamma": 0.99,
    "explore_step": 450000,
    "buffer_size": 50000,
    "batch_size": 32,
    "start_train_step": 25000,
    "target_update_period": 1000,
}

train = {
    "training" : True,
    "load_path" : None,
    "train_step" : 500000,
    "test_step" : 50000,
    "print_period" : 10,
    "save_period" : 500,
    "test_iteration": 10,
}