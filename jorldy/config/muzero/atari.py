### MuZero Atari Config ###

env = {
    # "name": it should be defined in the command. ex) python main.py --config config.AGENT.atari --env.name breakout
    "render": False,
    "gray_img": True,
    "img_width": 96,
    "img_height": 96,
    "stack_frame": 1,
    "no_op": True,
    "skip_frame": 4,
    "reward_clip": True,
    "episodic_life": True,
}

agent = {
    "name": "muzero",
    "network": "muzero_resnet",
    "head": "mlp",
    "hidden_size": 128,
    "gamma": 0.997,
    "buffer_size": 300000,
    "batch_size": 32,
    "v_boundary": {
        'min': -10.0,
        'max': 10.0,
    },
    "r_boundary": {
        'min': -1.0,
        'max': 1.0,
    },
    "num_support": {
        "v_support": 51,
        "r_support": 21,
    },
    "start_train_step": 3000,
    "policy_train_delay": 20000,
    "max_trajectory_size": 200,
    "value_loss_weight": 1.0,
    "num_unroll": 5,
    "num_td_step": 5,
    "num_stack": 4,
    "num_rb": 2,
    # PER
    "alpha": 1.0,
    "beta": 1.0,
    "uniform_sample_prob": 1e-3,
    # MCTS
    "num_mcts": 50,
    "num_eval_mcts": 30,
    "mcts_alpha_max": 1.0,
    "mcts_alpha_min": 0.3,
    # Optional Feature
    "use_prev_rand_action": True,
    "use_over_rand_action": True,
    "use_uniform_policy": False,
    "use_ssc_loss": False,
}

optim = {
    "name": "adam",
    "weight_decay": 1e-4,
    "lr": 1e-4,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 10000000,
    "print_period": 1000,
    "save_period": 100000,
    "eval_iteration": 2,
    "record": True,
    "record_period": 50000,
    # distributed setting
    "distributed_batch_size": 512,
    "update_period": 200,
    "num_workers": 64,
}
