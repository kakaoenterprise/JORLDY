### Munchausen DQN Super Mario Bros Config ###

env = {
    "name": "super_mario_bros",
    "render": False,
    "gray_img": True,
    "img_width": 84,
    "img_height": 84,
    "stack_frame": 4,
}

agent = {
    "name": "m_dqn",
    "network": "discrete_q_network",
    "head": "cnn",
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.01,
    "explore_ratio": 0.1,
    "buffer_size": 1000000,
    "batch_size": 32,
    "start_train_step": 100000,
    "target_update_period": 10000,
    "lr_decay": True,
    # M-DQN Parameters
    "alpha": 0.9,
    "tau": 0.03,
    "l_0": -1,
}

optim = {
    "name": "adam",
    "lr": 1e-4,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 100000000,
    "print_period": 5000,
    "save_period": 50000,
    "eval_iteration": 1,
    "record": True,
    "record_period": 200000,
    # distributed setting
    "update_period": 32,
    "num_workers": 16,
}
