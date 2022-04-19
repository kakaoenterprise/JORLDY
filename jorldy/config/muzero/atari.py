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
    "head": "residualblock",
    "gamma": 0.997,
    "buffer_size": 1250000,
    "batch_size": 32,
    "start_train_step": 10000,
    "policy_train_delay": 5000,
    "max_trajectory_size": 200,
    "value_loss_weight": 1.0,
    "num_unroll": 5,
    "num_td_step": 10,
    "num_stack": 32,
    "num_rb": 16,
    # out of range state setting
    "enable_after_random_action": True,
    "enable_prev_random_action": False,
    "enable_uniform_policy": True,
    # PER
    "alpha": 0.6,
    "beta": 0.4,
    "uniform_sample_prob": 1e-3,
    # MCTS
    "num_mcts": 30,
    "num_eval_mcts": 15,
    # Self Supervised Consistency Loss
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
    "run_step": 100000000,
    "print_period": 2,
    "save_period": 100000,
    "eval_iteration": 5,
    "record": True,
    "record_period": 500000,
    # distributed setting
    "update_period": 200,
    "num_workers": 32,
}
