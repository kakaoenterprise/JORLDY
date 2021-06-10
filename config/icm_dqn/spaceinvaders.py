### ICM DQN SpaceInvaders Config ###

env = {
    "name": "spaceinvaders",
    "render": False,
    "gray_img": True,
    "img_width": 84,
    "img_height": 84,
    "stack_frame": 4,
}

agent = {
    "name": "icm_dqn",
    "network": "dqn_cnn",
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "explore_step": 1000000,
    "buffer_size": 1000000,
    "batch_size": 32,
    "start_train_step": 100000,
    "target_update_period": 10000,
    # Parameters for Curiosity-driven Exploration
    "icm_network": "icm_cnn",
    "action_type": "discrete",
    "beta": 0.2,
    "lamb": 1.0,
    "eta": 0.01,
    "extrinsic_coeff": 1.0,
    "intrinsic_coeff": 0.01,
}

train = {
    "training" : True,
    "load_path" : None,
    "run_step" : 100000000,
    "print_period" : 5000,
    "save_period" : 50000,
    "test_iteration": 5,
    # distributed setting
    "update_period" : 32,
    "num_worker" : 16,
}