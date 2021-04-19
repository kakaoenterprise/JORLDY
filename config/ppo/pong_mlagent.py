### PPO Pong_ML-Agents Config ###

env = {
    "name": "pong_mlagent",
    "train_mode": True
}

agent = {
    "name": "ppo",
    "network": "discrete_pi_v",
    "optimizer": "adam",
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "batch_size":64,
    "n_step": 500,
    "n_epoch": 3,
    "_lambda": 0.95,
    "epsilon_clip": 0.2,
    "vf_coef": 0.5,
    "ent_coef": 0.0,
}

train = {
    "training" : True,
    "load_path" : None,
    "train_step" : 500000,
    "test_step" : 50000,
    "print_term" : 10,
    "save_term" : 500,
    "test_iteration": 5,
}