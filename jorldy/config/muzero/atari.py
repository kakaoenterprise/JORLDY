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
    "reward_clip": False,
    "episodic_life": True,
}

agent = {
    "name": "muzero",
    "network": "muzero_resnet",
    "head": "residualblock",
    "hidden_size": 64,
    "gamma": 0.997,
    "buffer_size": 1000000,
    "batch_size": 32,
    "num_support": 10,
    "start_train_step": 10000,
    "policy_train_delay": 10000,
    "max_trajectory_size": 200,
    "value_loss_weight": 1.0,
    "num_unroll": 5,
    "num_td_step": 5,
    "num_stack": 4,
    "num_rb": 1,
    # PER
    "alpha": 1.0,
    "beta": 1.0,
    "uniform_sample_prob": 1e-3,
    # MCTS
    "num_mcts": 30,
    "num_eval_mcts": 15,
    "mcts_alpha_max": 0.4,
    "mcts_alpha_min": 0.0,
    # Optional Feature
    "use_prev_rand_action": True,
    "use_over_rand_action": True,
    "use_uniform_policy": True,
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
    "record_period": 500000,
    # distributed setting
    "distributed_batch_size": 512,
    "update_period": 200,
    "num_workers": 64,
}
