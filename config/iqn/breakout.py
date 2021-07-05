### IQN BreakOut Config ###

env = {
    "name": "breakout",
    "render": False,
    "gray_img": True,
    "img_width": 84,
    "img_height": 84,
    "stack_frame": 4,
}

agent = {
    "name": "iqn",
    "network": "iqn_cnn",
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
    "target_update_period": 10000,
    
    "num_sample": 64,
    "embedding_dim": 64,
    "sample_min": 0.0,
    "sample_max": 1.0,
}

train = {
    "training" : True,
    "load_path" : None,
    "run_step" : 30000000,
    "print_period" : 50000,
    "save_period" : 500000,
    "test_iteration": 5,
    "record": True,
    "record_period": 300000,
    # distributed setting
    "update_period" : 8,
    "num_worker" : 16,
}
