### Ape-X Atari Config ###

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
    "name": "apex",
    "network": "dueling_cnn",
    "optimizer": "rmsprop",
    "opt_eps": 1.5e-7,
    "learning_rate": 0.00025/4,
    "gamma": 0.99,
    "buffer_size": 1000000,
    "batch_size": 32,
    "start_train_step": 50000,
    "target_update_period": 2500,
    # MultiStep
    "n_step": 3,
    # PER
    "alpha": 0.6,
    "beta": 0.4,
    "uniform_sample_prob": 1e-3,
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
    "distributed_batch_size" : 512,
    "update_period" : 50,
    "num_worker" : 32,
}