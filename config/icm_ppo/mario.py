### ICM PPO Mario Config ###

env = {
    "name": "mario",
    "render": False,
    "gray_img": True,
    "img_width": 84,
    "img_height": 84,
    "stack_frame": 4,
#     "no_op": True,
    "reward_clip": True,
    "dead_penalty": True,
}

agent = {
    "name":"icm_ppo",
    "network":"discrete_pi_v_cnn",
    "gamma":0.99,
    "batch_size":32,
    "n_step": 128,
    "n_epoch": 3,
    "_lambda": 0.95,
    "epsilon_clip": 0.1,
    "vf_coef": 1.0,
    "ent_coef": 0.1,
    "clip_grad_norm": 1.0,
    # Parameters for Curiosity-driven Exploration
    "icm_network": "icm_cnn",
    "beta": 0.2,
    "lamb": 1.0,
    "eta": 0.1,
    "extrinsic_coeff": 1.0,
    "intrinsic_coeff": 1.0,
}

optim = {
    "name":"adam",
    "lr": 2.5e-4,
}

train = {
    "training" : True,
    "load_path" : None, #"./logs/mario/icm_ppo/20210729225927/",#
    "run_step" : 30000000,
    "print_period" : 50000,
    "save_period" : 500000,
    "test_iteration": 1,
    "record": True,
    "record_period": 300000,
    # distributed setting
    "distributed_batch_size": 1024,
    "update_period" : agent["n_step"],
    "num_worker" : 32,
}
