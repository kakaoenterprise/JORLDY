### C51 Pong_ML-Agents Config ###

env = {
    "name": "pong_mlagent",
    "train_mode": True
}

agent = {
    "name": "c51",
    "network": "dqn",
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.1,
    "explore_step": 450000,
    "buffer_size": 50000,
    "batch_size": 64,
    "start_train_step": 25000,
    "target_update_term": 1000,
    "v_min": -10,
    "v_max": 10,
    "num_support": 51
}

train = {
    "training" : True,
    "load_path" : None,
    "train_step" : 500000,
    "test_step" : 50000,
    "print_term" : 10,
    "save_term" : 500,
}