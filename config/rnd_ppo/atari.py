### RND PPO Atari Config ###

env = {
    # "name": it should be defined in the command. ex) python main.py --config config.dqn.atari --env.name breakout
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
    "name":"rnd_ppo",
    "network":"discrete_pi_v_cnn",
    "optimizer":"adam",
    "learning_rate": 0.0001,
    "gamma":0.999,
    "batch_size":32,
    "n_step": 128,
    "n_epoch": 3,
    "_lambda": 0.95,
    "epsilon_clip": 0.1,
    "vf_coef": 1.0,
    "ent_coef": 0.001,
    "clip_grad_norm": 1.0,
    "use_standardization": False,
    # Parameters for Random Network Distillation
    "rnd_network": "rnd_cnn",
    "gamma_i": 0.99,
    "extrinsic_coeff": 2.0,
    "intrinsic_coeff": 1.0,
    "obs_normalize": True,
    "ri_normalize": True,
    "batch_norm": True,
}

train = {
    "training" : True,
    "load_path" : None,
    "run_step" : 100000000,
    "print_period" : 10000,
    "save_period" : 100000,
    "test_iteration": 1,
    "record" : True,
    "record_period" : 1000000,
    # distributed setting
    "update_period" : agent["n_step"],
    "num_worker" : 127,
}
