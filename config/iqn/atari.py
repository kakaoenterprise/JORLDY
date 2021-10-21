### IQN Atari Config ###

env = {
    # "name": it should be defined in the command. ex) python main.py --config config.AGENT.atari --env.name breakout
    "render": False,
    "gray_img": True,
    "img_width": 84,
    "img_height": 84,
    "stack_frame": 4,
    "no_op": False,
    "reward_clip": True,
    "dead_penalty": True,
}

agent = {
    "name": "iqn",
    "network": "iqn",
    "head": "cnn",
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.1,
    "explore_step": 1000000,
    "buffer_size": 1000000,
    "batch_size": 32,
    "start_train_step": 100000,
    "target_update_period": 10000,
    
    "num_sample": 64,
    "embedding_dim": 64,
    "sample_min": 0.0,
    "sample_max": 1.0,
}

optim = {
    "name": "adam",
    "eps": 1e-2/agent['batch_size'],
    "lr": 5e-5,
}

train = {
    "training" : True,
    "load_path" : None,
    "run_step" : 30000000,
    "print_period" : 10000,
    "save_period" : 100000,
    "eval_iteration": 5,
    "record" : True,
    "record_period" : 300000,
    # distributed setting
    "update_period" : 32,
    "num_workers" : 16,
}