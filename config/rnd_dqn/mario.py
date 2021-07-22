### RND DQN Mario Config ###

env = {
    "name": "mario",
    "render": False,
    "gray_img": True,
    "img_width": 84,
    "img_height": 84,
    "stack_frame": 4,
}

agent = {
    "name": "rnd_dqn",
    "network": "dqn_cnn",
    "optimizer": "adam",
    "learning_rate": 1e-4, #0.00025,
    "gamma": 0.99,
    "explore_step": 1000000,
    "buffer_size": 1000000,
    "batch_size": 32,
    "start_train_step": 50000,
    "target_update_period": 10000,
    # Parameters for Random Network Distillation
    "rnd_network": "rnd_cnn",
    "gamma_i": 0.99,
    "extrinsic_coeff": 1.0,
    "intrinsic_coeff": 1.0,
    "obs_normalize": False,
    "ri_normalize": False,
}

train = {
    "training" : True,
    "load_path" : './logs/mario/rnd_dqn/20210721143654',
    "run_step" : 100000000,
    "print_period" : 5000,
    "save_period" : 50000,
    "test_iteration": 1,
    "record" : True,
    "record_period" : 200000,
    # distributed setting
    "update_period" : 32,
    "num_worker" : 16,
}
