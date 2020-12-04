### SAC CartPole Config ###

env = {
    "name":"cartpole",
    "mode":"continuous",
    "render":False,
}

agent = {
    "name":"sac",
    "actor":"sac_actor",
    "critic":"sac_critic",
    "actor_optimizer":"adam",
    "critic_optimizer":"adam",
    "alpha_optimizer":"adam",
    "actor_lr":5e-4,
    "critic_lr":1e-3,
    "alpha_lr":3e-4,
    "use_dynamic_alpha":True,
    "gamma":0.99,
    "tau":5e-3,
    "buffer_size":50000,
    "batch_size":64,
    "start_train_step":5000,
    "static_log_alpha":-2.0,
}

train = {
    "training" : True,
    "load_path" : None, #"./logs/cartpole/sac/20201204202618/",
    "train_step" : 80000,
    "test_step" : 50000,
    "print_term" : 10,
    "save_term" : 100,
}