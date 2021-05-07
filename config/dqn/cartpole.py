### DQN CratPole Config ###

env = {
    "name":"cartpole",
    "mode":"discrete",
    "render":False,
}

agent = {
    "name": "dqn",
    "network": "dqn",
    "optimizer": "adam",
    "learning_rate": 0.0005,
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.01,
    "explore_step": 20000,
    "buffer_size": 50000,
    "batch_size": 32,
    "start_train_step": 2000,
    "target_update_term": 1000,
}

train = {
    "training" : True,
    "load_path" : None,
    "train_step" : 80000,
    "test_step" : 30000,
    "print_term" : 20,
    "save_term" : 1000,
    "test_iteration": 10,
}
