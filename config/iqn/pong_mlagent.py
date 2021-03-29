### IQN Pong_ML-Agents Config ###

env = {
    "name": "pong_mlagent",
    "train_mode": True
}

agent = {
    "name": "iqn",
    "network": "iqn",
    "optimizer": "adam",
    "opt_eps": 1e-2/32,
    "learning_rate": 0.00005,
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.1,
    "explore_step": 450000,
    "buffer_size": 50000,
    "batch_size": 32,
    "start_train_step": 25000,
    "target_update_term": 1000,
    
    "num_sample": 64,
    "embedding_dim": 64,
    "sample_min": 0.75, #0,
    "sample_max": 1.0
}

train = {
    "training" : False, #True,
    "load_path" : "logs/pong_mlagent/iqn/20210322152835",#None,
    "train_step" : 500000,
    "test_step" : 500000,
    "print_term" : 10,
    "save_term" : 500,
    "test_iteration": 10,
}