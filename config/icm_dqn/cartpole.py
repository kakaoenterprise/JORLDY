### ICM DQN CratPole Config ###

env = {
    "name":"cartpole",
    "mode":"discrete",
    "render":False,
}

agent = {
    "name": "icm_dqn",
    "network": "dqn",
    "optimizer": "adam",
    "learning_rate": 0.0005,
    "gamma": 0.99,
    "explore_step": 20000,
    "buffer_size": 50000,
    "batch_size": 32,
    "start_train_step": 2000,
    "target_update_period": 1000,
    # Parameters for Curiosity-driven Exploration
    "icm_network": "icm",
    "action_type": "discrete",
    "beta": 0.2,
    "lamb": 1.0,
    "eta": 0.01,
    "extrinsic_coeff": 1.0,
    "intrinsic_coeff": 0.01,
}

train = {
    "training" : True,
    "load_path" : None,
    "run_step" : 100000,
    "print_period" : 1000,
    "save_period" : 10000,
    "test_iteration": 10,
    # distributed setting
    "update_period" : 32,
    "num_worker" : 8,
}
