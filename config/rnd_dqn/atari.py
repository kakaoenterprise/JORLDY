### RND DQN Atari Config ###

env = {
    # "name": it should be defined in the command. ex) python main.py --config config.dqn.atari --env.name breakout
    "render": False,
    "gray_img": True,
    "img_width": 84,
    "img_height": 84,
    "stack_frame": 4,
    "no_op": True,
    "reward_clip": True,
}

agent = {
    "name": "rnd_dqn",
    "network": "dqn_cnn",
    "optimizer": "rmsprop",
    "learning_rate": 2.5e-4,
    "gamma": 0.999,
    "explore_step": 1000000,
    "buffer_size": 1000000,
    "batch_size": 32,
    "start_train_step": 100000,
    "target_update_period": 10000,
    # Parameters for Random Network Distillation
    "rnd_network": "rnd_cnn",
    "gamma_i": 0.99,
    "extrinsic_coeff": 0.0,
    "intrinsic_coeff": 1.0,
}

train = {
    "training" : True,
    "load_path" : None,
    "run_step" : 30000000,
    "print_period" : 10000,
    "save_period" : 100000,
    "test_iteration": 5,
    "record" : True,
    "record_period" : 300000,
    # distributed setting
    "update_period" : 32,
    "num_worker" : 16,
}