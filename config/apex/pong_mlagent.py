### Rainbow DQN Pong_ML-Agents Config ###

env = {
    "name": "pong_mlagent",
    "train_mode": True
}

agent = {
    "name": "apex",
    "network": "dueling",
    "optimizer": "rmsprop",
    "opt_eps": 1.5e-7,
    "learning_rate": 0.00025/4,
    "gamma": 0.99,
    "buffer_size": 50000,
    "batch_size": 32,
    "start_train_step": 25000,
    "target_update_period": 1000,
    # MultiStep
    "n_step": 3,
    # PER
    "alpha": 0.6,
    "beta": 0.4,
    "uniform_sample_prob": 1e-3,
}

train = {
    "training" : True,
    "load_path" : None,
    "run_step" : 200000,
    "print_period" : 5000,
    "save_period" : 50000,
    "test_iteration": 10,
    # distributed setting
    "distributed_batch_size" : 512,
    "update_period" : 16,
    "num_worker" : 16,
}