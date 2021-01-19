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
    "learning_rate": 1e-2,
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.1,
    "explore_step": 1000000,
    "buffer_size": 1000000,
    "batch_size": 32,
    "start_train_step": 100000,
    "target_update_term": 10000,
    "v_min": -10,
    "v_max": 10,
    "num_support": 51
}

train = {
    "training" : True,
    "load_path" : None,
    "train_step" : 20000000,
    "test_step" : 1000000,
    "print_term" : 50,
    "save_term" : 500,
}