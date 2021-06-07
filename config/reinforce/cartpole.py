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
    "run_step" : 100000,
    "print_period" : 1000,
    "save_period" : 10000,
    "test_iteration": 10,
}