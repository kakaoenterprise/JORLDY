env = {
    "name": "hopper_mlagent",
    "train_mode": True
}

agent = {
    "name":"sac",
    "actor":"sac_actor",
    "critic":"sac_critic",
    "actor_optimizer":"adam",
    "critic_optimizer":"adam",
    "alpha_optimizer":"adam",
    "actor_lr":3e-4,
    "critic_lr":3e-4,
    "alpha_lr":3e-4,
    "use_dynamic_alpha":True,
    "gamma":0.99,
    "tau":5e-3,
    "buffer_size":50000,
    "batch_size":128,
    "start_train_step":25000,
    "static_log_alpha":-2.0,
}


train = {
    "training" : True,
    "load_path" : None, 
    "train_step" : 1000000,
    "test_step" : 100000,
    "print_term" : 10,
    "save_term" : 1000,
    "test_iteration": 10,
}