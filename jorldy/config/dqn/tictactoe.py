### DQN TicTacToe Config ###

env = {
    "name": "tictactoe", "input_type": "vector",
}

agent = {
    "name": "dqn",
    "network": "discrete_q_network",
    "head": "mlp",
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.01,
    "explore_ratio": 0.2,
    "buffer_size": 1000,
    "batch_size": 16,
    "start_train_step": 200,
    "target_update_period": 50,
    "lr_decay": True,
}

optim = {
    "name": "adam",
    "lr": 0.0001,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 10000,
    "print_period": 100,
    "save_period": 1000,
    "eval_iteration": 10,
    # distributed setting
    "update_period": 32,
    "num_workers": 8,
}
