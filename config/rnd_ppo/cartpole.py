### RND PPO CartPole Config ###

env = {
    "name":"cartpole",
    "mode": "discrete",
    "render":False,
}

agent = {
    "name":"ppo",
    "network":"discrete_pi_v",
    "optimizer":"adam",
    "learning_rate": 3e-4,
    "gamma":0.99,
    "batch_size":64,
    "n_step": 128,
    "n_epoch": 3,
    "_lambda": 0.95,
    "epsilon_clip": 0.1,
    "vf_coef": 0.5,
    "ent_coef": 0.0,
    # Parameters for Random Network Distillation
    "rnd_network": "rnd",
    "gamma_i": 0.99
    "extrinsic_coeff": 0.0,
    "intrinsic_coeff": 1.0,
}

train = {
    "training" : True,
    "load_path" : None,
    "run_step" : 100000,
    "print_period" : 1000,
    "save_period" : 10000,
    "test_iteration": 10,
    # distributed setting
    "update_period" : agent["n_step"],
    "num_worker" : 8,
}
