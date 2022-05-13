### MuZero Pong_ML-Agents Config ###

env = {"name": "pong_mlagent", "time_scale": 12.0}

agent = {
    "name": "muzero",
    "network": "muzero_mlp",
    "head": "mlp",
    "hidden_size": 64,
    "gamma": 0.997,
    "buffer_size": 100000,
    "batch_size": 32,
    "num_support": 10,
    "start_train_step": 3000,
    "policy_train_delay": 5000,
    "max_trajectory_size": 1000,
    "value_loss_weight": 1.0,
    "num_unroll": 5,
    "num_td_step": 5,
    "num_stack": 1,
    "num_rb": 1,
    "lr_decay": False,
    # PER
    "alpha": 0.6,
    "beta": 0.4,
    "uniform_sample_prob": 1e-3,
    # MCTS
    "num_mcts": 30,
    "num_eval_mcts": 15,
    "mcts_alpha_max": 1.0,
    "mcts_alpha_min": 0.1,
    # Optional Feature
    "use_prev_rand_action": True,
    "use_over_rand_action": True,
    "use_uniform_policy": True,
    "use_ssc_loss": False,
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
    "print_period": 5000,
    "save_period": 50000,
    "eval_iteration": 3,
    # distributed setting
    "distributed_batch_size": 128,
    "update_period": 200,
    "num_workers": 32,
}
