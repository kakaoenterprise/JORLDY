### IQN CratPole Config ###

env = {
    "name":"cartpole",
    "mode":"discrete",
    "render":False,
}

agent = {
    "name": "iqn",
    "network": "iqn",
    "optimizer": "adam",
    "opt_eps": 0.01/64,
    "learning_rate": 0.00005,
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.1,
    "explore_step": 20000,
    "buffer_size": 10000,
    "batch_size": 64,
    "start_train_step": 10000,
    "target_update_term": 500,
    
    "num_sample": 32,
    "embedding_dim": 64,
    "sample_min": 0.0,
    "sample_max": 1.0
}

train = {
    "training" : True,
    "load_path" : None,
    "train_step" : 50000,
    "test_step" : 10000,
    "print_term" : 5,
    "save_term" : 1000,
}
