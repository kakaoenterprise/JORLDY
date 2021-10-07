### RND PPO Mario Config ###

env = {
    "name": "mario",
    "render": False,
    "gray_img": True,
    "img_width": 84,
    "img_height": 84,
    "stack_frame": 4,
    "no_op": False,
    "reward_clip": True,
    "dead_penalty": False,
}

agent = {
    "name":"rnd_ppo",
    "network":"discrete_policy_value",
    "head": "cnn",
    "gamma":0.99,
    "batch_size":8,
    "n_step": 128,
    "n_epoch": 4,
    "_lambda": 0.95,
    "epsilon_clip": 0.1,
    "vf_coef": 1.0,
    "ent_coef": 0.0001,
    "clip_grad_norm": 1.0,
    "use_standardization": False,
    # Parameters for Random Network Distillation
    "rnd_network": "rnd_cnn_rnn", # rnd_mlp, rnd_cnn, rnd_mlp_rnn, rnd_cnn_rnn
    "gamma_i": 0.99,
    "extrinsic_coeff": 1.0,
    "intrinsic_coeff": 1.0,
    "obs_normalize": True,
    "ri_normalize": True,
    "batch_norm": True,
    "seq_len": 8, # for "rnn"
}

optim = {
    "name":"adam",
    "lr": 0.0001,
}

train = {
    "training" : True,
    "load_path" : None,
    "run_step" : 30000000,
    "print_period" : 10000,
    "save_period" : 100000,
    "test_iteration": 1,
    "record" : True,
    "record_period" : 250000,
    # distributed setting
    "distributed_batch_size" : 256,
    "update_period" : agent["n_step"],
    "num_workers" : 64,
}