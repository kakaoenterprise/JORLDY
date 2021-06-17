### ICM PPO Pendulum Config ###

env = {
    "name":"pendulum",
    "render":False,
}

agent = {
    "name": "icm_ppo",
    "network": "continuous_pi_v",
    "optimizer": "adam",
    "learning_rate": 5e-4,
    "gamma": 0.99,
    "batch_size":64,
    "n_step": 200,
    "n_epoch": 3,
    "_lambda": 0.95,
    "epsilon_clip": 0.1,
    "vf_coef": 0.5,
    "ent_coef": 0.0,
    # Parameters for Curiosity-driven Exploration
    "icm_network": "icm",
    "beta": 0.2,
    "lamb": 1.0,
    "eta": 0.01,
    "extrinsic_coeff": 1.0,
    "intrinsic_coeff": 0.01,
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
