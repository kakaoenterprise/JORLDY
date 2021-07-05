### PPO Mario Config ###

env = {
    "name": "mario",
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
    "batch_size":64,
    "n_step": 2048,
    "n_epoch": 3,
    "_lambda": 0.95,
    "epsilon_clip": 0.1,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
}

train = {
    "training" : True,
    "load_path" : None,
    "run_step" : 100000000,
    "print_period" : 50000,
    "save_period" : 500000,
    "test_iteration": 1,
    "record": True,
    "record_period" : 1000000,
    # distributed setting
    "update_period" : agent["n_step"],
    "num_worker" : 16,
}
