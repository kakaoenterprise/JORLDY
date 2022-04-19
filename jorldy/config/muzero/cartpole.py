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
    "buffer_size": 3000,
    "batch_size": 32,
    "num_support": 10,
    "start_train_step": 1000,
    "policy_train_delay": 2000,
    "max_trajectory_size": 1000,
    "value_loss_weight": 1.0,
    "num_unroll": 5,
    "num_td_step": 5,
    "num_stack": 4,
    "num_rb": 1,
    "lr_decay": False,
    # out of range state setting
    "enable_after_random_action": True,
    "enable_prev_random_action": False,
    "enable_uniform_policy": True,
    # MCTS
    "num_mcts": 30,
    "num_eval_mcts": 15,
    # Self Supervised Consistency Loss
    "use_ssc_loss": True, 
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
