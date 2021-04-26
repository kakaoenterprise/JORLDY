### PPO Asterix Config ###

env = {
    "name": "asterix",
    "render": False,
    "gray_img": True,
    "img_width": 84,
    "img_height": 84,
    "stack_frame": 4,
}

agent = {
    "name":"ppo",
    "network":"discrete_pi_v_cnn",
    "optimizer":"adam",
    "learning_rate": 3e-4,
    "gamma":0.99,
    "batch_size":64,
    "n_step": 500,
    "n_epoch": 10,
    "_lambda": 0.95,
    "epsilon_clip": 0.1,
}

train = {
    "training" : True,
    "load_path" : None,
    "train_step" : 20000000,
    "test_step" : 1000000,
    "print_term" : 50,
    "save_term" : 500,
    "test_iteration": 10,
}