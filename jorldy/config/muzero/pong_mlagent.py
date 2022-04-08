### MuZero Pong_ML-Agents Config ###

env = {"name": "pong_mlagent", "time_scale": 12.0}

agent = {
    "name": "muzero",
    "network": "muzero_mlp",
    "head": "mlp_residualblock",
    "hidden_size": 128,
    "gamma": 0.997,
    "buffer_size": 50000,
    "batch_size": 32,
    "num_support": 100,
    "start_train_step": 1000,
    "max_trajectory_size": 50,
    "value_loss_weight": 1.0,
    "num_unroll": 5,
    "num_td_step": 5,
    "num_stack": 2,
    "num_rb": 1,
    "lr_decay": False,
    "num_mcts": 30,
    "num_eval_mcts": 15,
}

optim = {
    "name": "adam",
    "weight_decay": 1e-4,
    "lr": 1e-3,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 200000,
    "print_period": 1000,
    "save_period": 50000,
    "eval_iteration": 3,
    # distributed setting
    "update_period": 8,
    "num_workers": 16,
}
