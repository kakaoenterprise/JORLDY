### ICM DQN Pong_ML-Agents Config ###

env = {
    "name": "pong_mlagent",
    "train_mode": True
}

agent = {
    "name": "icm_dqn",
    "network": "dqn",
    "optimizer": "adam",
    "learning_rate": 0.00025,
    "gamma": 0.99,
    "explore_step": 450000,
    "buffer_size": 50000,
    "batch_size": 32,
    "start_train_step": 25000,
    "target_update_period": 1000,
    # Parameters for Curiosity-driven Exploration
    "icm_network": "icm",
    "beta": 0.2,
    "lamb": 1.0,
    "eta": 0.01,
    "extrinsic_coeff": 1.0,
    "intrinsic_coeff": 0.001,
}

train = {
    "training" : True,
    "load_path" : None,
    "run_step" : 200000,
    "print_period" : 5000,
    "save_period" : 50000,
    "test_iteration": 10,
    # distributed setting
    "update_period" : 8,
    "num_worker" : 16,
}