### Multistep DQN Procgen Config ###

env = {
    # "name": it should be defined in the command. ex) python main.py --config config.AGENT.procgen --env.name coinrun
    "render": False,
    "gray_img": True,
    "stack_frame": 4,
    "no_op": True,
    "skip_frame": 4,
    "reward_clip": True,
}

agent = {
    "name": "multistep",
    "network": "discrete_q_network",
    "head": "cnn",
    "gamma": 0.99,
    "n_step": 4,
    "epsilon_init": 1.0,
    "epsilon_min": 0.1,
    "explore_ratio": 0.1,
    "buffer_size": 1000000,
    "batch_size": 32,
    "start_train_step": 100000,
    "target_update_period": 10000,
    "lr_decay": True,
}

optim = {
    "name": "adam",
    "lr": 2.5e-4,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 30000000,
    "print_period": 10000,
    "save_period": 100000,
    "eval_iteration": 5,
    "record": True,
    "record_period": 300000,
    # distributed setting
    "update_period": 32,
    "num_workers": 16,
}
