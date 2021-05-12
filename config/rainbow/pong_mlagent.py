### Rainbow DQN Pong_ML-Agents Config ###

env = {
    "name": "pong_mlagent",
    "train_mode": True
}

agent = {
    "name": "rainbow",
    "network": "rainbow",
    "optimizer": "adam",
    "learning_rate": 0.0000625,
    "gamma": 0.99,
    "explore_step": 450000,
    "buffer_size": 50000,
    "batch_size": 32,
    "start_train_step": 25000,
    "target_update_period": 1000,
    # MultiStep
    "n_step": 4,
    # PER
    "alpha": 0.6,
    "beta": 0.4,
    "learn_period": 4,
    "uniform_sample_prob": 1e-3,
    # C51
    "v_min": -10,
    "v_max": 10,
    "num_support": 51
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