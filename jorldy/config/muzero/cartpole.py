### MuZero CartPole Config ###

env = {
    "name": "cartpole",
    "action_type": "discrete",
    "render": False,
}

agent = {
    "name": "muzero",
    "network": "muzero_mlp",
    "head": "mlp",
    "gamma": 0.997,
    "epsilon_init": 1.0,
    "epsilon_min": 0.01,
    "explore_ratio": 0.2,
    "buffer_size": 50000,
    "batch_size": 128,
    "num_support": 10,
    "start_train_step": 1000,
    "trajectory_size": 1000,
    "num_simulation": 50,
    "value_loss_weight": 1.0,
    "num_unroll": 5,
    "num_td_step": 8,
    "num_stack": 8,
}

optim = {
    "name": "adam",
    "lr": 0.0001,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 100000,
    "print_period": 1000,
    "save_period": 10000,
    "eval_iteration": 10,
    # distributed setting
    "update_period": 32,
    "num_workers": 8,
}
