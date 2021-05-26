### Rainbow IQN SpaceInvaders Config ###

env = {
    "name": "spaceinvaders",
    "render": False,
    "gray_img": True,
    "img_width": 80,
    "img_height": 80,
    "stack_frame": 4,
}

agent = {
    "name": "rainbow_iqn",
    "network": "rainbow_iqn_cnn",
    "optimizer": "adam",
    "learning_rate": 0.0000625,
    "gamma": 0.99,
    "explore_step": 1000000,
    "buffer_size": 100000,
    "batch_size": 64,
    "start_train_step": 100000,
    "target_update_period": 500,
    # MultiStep
    "n_step": 3,
    # PER
    "alpha": 0.6,
    "beta": 0.4,
    "learn_period": 4,
    "uniform_sample_prob": 1e-3,
    # IQN
    "num_sample": 64,
    "embedding_dim": 64,
    "sample_min": 0.0,
    "sample_max": 1.0,
}

train = {
    "training" : True,
    "load_path" : None,
    "train_step" : 10000000,
    "test_step" : 1000000,
    "print_period" : 10,
    "save_period" : 100,
    "test_iteration": 10,
}