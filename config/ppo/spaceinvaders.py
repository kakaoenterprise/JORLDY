### PPO SpaceInvaders Config ###

env = {
    "name": "spaceinvaders",
    "render": False,
    "gray_img": True,
    "img_width": 80,
    "img_height": 80,
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
    "load_path" : None, #"./logs/breakout/dqn/20201027142347/",
    "train_step" : 10000000,
    "test_step" : 1000000,
    "print_term" : 10,
    "save_term" : 100,
    "test_iteration": 10,
}