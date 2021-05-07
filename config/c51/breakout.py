### C51 BreakOut Config ###

env = {
    "name": "breakout",
    "render": False,
    "gray_img": True,
    "img_width": 84,
    "img_height": 84,
    "stack_frame": 4,
}

agent = {
    "name": "c51",
    "network": "dqn_cnn",
    "optimizer": "adam",
    "opt_eps": 1e-2/32,
    "learning_rate": 1e-4, #0.00025,
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.01,
    "explore_step": 1000000,
    "buffer_size": 1000000,
    "batch_size": 32,
    "start_train_step": 100000,
    "target_update_period": 10000,
    "v_min": -1,
    "v_max": 10,
    "num_support": 51
}

train = {
    "training" : True, #False,
    "load_path" : None, #"./logs/breakout/c51/20210225183508",
    "train_step" : 100000000,
    "test_step" : 1000000,
    "print_period" : 50,
    "save_period" : 500,
    "test_iteration": 5,
}