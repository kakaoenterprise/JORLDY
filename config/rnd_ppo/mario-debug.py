### RND PPO Mario Config ###

env = {
    "name": "mario",
    "render": False,
    "gray_img": True,
    "img_width": 84,
    "img_height": 84,
    "stack_frame": 4,
#     "no_op": True,
    "reward_clip": True,
    "dead_penalty": False,
}

agent = {
    "name":"rnd_ppo",
    "network":"discrete_pi_v_cnn",
    "optimizer":"adam",
    "learning_rate": 2.5e-4,
    "gamma":0.99,
    "batch_size":32,
    "n_step": 128,
    "n_epoch": 3,
    "_lambda": 0.95,
    "epsilon_clip": 0.1,
    "vf_coef": 1.0,
    "ent_coef": 0.01,
    "clip_grad_norm": 1.0,
    # Parameters for Random Network Distillation
    "rnd_network": "rnd_cnn",
    "gamma_i": 0.99,
    "extrinsic_coeff": 1.0,
    "intrinsic_coeff": 10.0,
    "obs_normalize": True,
    "ri_normalize": True,
#     "ri_normalize": False,
}

train = {
    "training" : True,
    "load_path" : "./logs/mario/rnd_ppo/20210805143041/",
    "run_step" : 30000000,
    "print_period" : 10000,
    "save_period" : 100000,
#     "save_period" : 10000,
    "test_iteration": 1,
    "record" : True,
    "record_period" : 200000,
    # distributed setting
    "distributed_batch_size" : 1024,
    "update_period" : agent["n_step"],
    "num_worker" : 32,
}