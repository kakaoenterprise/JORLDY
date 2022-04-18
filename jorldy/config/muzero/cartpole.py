### MuZero CartPole Config ###

env = {
    "name": "cartpole",
    "action_type": "discrete",
    "render": False,
}

agent = {
    "name": "muzero",
    "network": "muzero_mlp",
    "head": "mlp_residualblock",
    "hidden_size": 64,
    "gamma": 0.99,
    "buffer_size": 5000,
    "batch_size": 32,
    "num_support": 10,
    "start_train_step": 1000,
    "max_trajectory_size": 100,
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
    "run_step": 100000,
    "print_period": 1000,
    "save_period": 10000,
    "eval_iteration": 3,
    # distributed setting
    "distributed_batch_size": 128,
    "update_period": 100,
    "num_workers": 32,
}