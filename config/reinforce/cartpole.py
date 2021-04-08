### REINFORCE CartPole Config ###

env = {
    "name":"cartpole",
    "mode": "discrete",
    "render":False,
}

agent = {
    "name":"reinforce",
    "network":"discrete_policy",
    "optimizer":"adam",
    "learning_rate": 1e-4,
    "gamma":0.99,
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