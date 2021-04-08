### PPO CartPole Config ###

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
    "n_step": 200,
    "n_epoch": 5,
    "_lambda": 0.95,
    "epsilon_clip": 0.1,
}

train = {
    "training" : True,
    "load_path" : None,
    "train_step" : 100000,
    "test_step" : 50000,
    "print_term" : 10,
    "save_term" : 100,
    "test_iteration": 10,
}