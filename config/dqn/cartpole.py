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
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.1,
    "explore_step": 10000,
    "buffer_size": 10000,
    "batch_size": 32,
    "start_train_step": 5000,
    "target_update_term": 200,
}

train = {
    "training" : True,
    "load_path" : None,
    "run_step" : 50000,
    "print_term" : 1000,
    "save_term" : 10000,
    "test_iteration": 5,
    "update_term" : 1,
    "num_worker" : 8,
}
