### PPO Pendulum Config ###

env = {
    "name":"pendulum",
    "render":False,
}

agent = {
    "name": "ppo",
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
}

train = {
    "training" : True,
    "load_path" : None,
    "run_step" : 100000,
    "print_period" : 10,
    "save_period" : 100,
    "test_iteration": 10,
    "test_iteration": 5,
    "update_term" : agent["n_step"],
    "num_worker" : 16,
}