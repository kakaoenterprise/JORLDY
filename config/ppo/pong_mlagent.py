### PPO Pong_ML-Agents Config ###

env = {
    "name": "pong_mlagent",
    "train_mode": True
}

agent = {
    "name": "ppo",
    "network": "discrete_pi_v",
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
    "run_step" : 500000,
    "print_term" : 1000,
    "save_term" : 10000,
    "test_iteration": 5,
    "update_term" : agent["n_step"],
    "num_worker" : 16,
}