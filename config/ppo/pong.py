### PPO Pong Config ###

env = {
    "name": "pong",
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
    "learning_rate": 2.5e-4,
    "gamma":0.99,
    "batch_size": 32,
    "n_step": 2048,
    "n_epoch": 3,
    "_lambda": 0.95,
    "epsilon_clip": 0.1,
    "vf_coef": 1.0,
    "ent_coef": 0.01,
}

train = {
    "training" : True,
    "load_path" : None,
    "run_step" : 10000000,
    "print_period" : 10000,
    "save_period" : 1000000,
    "test_iteration": 10,
    "record" : True,
    "record_period" : 50000,
    # distributed setting
    "update_period" : agent["n_step"],
    "num_worker" : 8,
}
