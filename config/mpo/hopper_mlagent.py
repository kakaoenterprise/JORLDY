### MPO Hopper Config ###

env = {
    "name": "hopper_mlagent",
    "train_mode": True
}

agent = {
    "name": "mpo",
    "actor": "continuous_policy",
    "critic": "ddpg_critic",
    "critic_loss_type": "retrace", # one of ['1-step TD', 'retrace']
    "gamma": 0.99,
    "buffer_size": 50000,
    "batch_size": 32,
    "n_step": 8,
    "start_train_step": 2000,
    "n_epoch": 64,
    "clip_grad_norm": 1.0,
    
    "min_eta": 1e-8,
    "min_alpha_mu": 1e-8,
    "min_alpha_sigma": 1e-8,
    
    "eps_eta": 0.01,
    "eps_alpha_mu": 0.01,
    "eps_alpha_sigma": 5*1e-5,
    
    "eta": 1.0,
    "alpha_mu": 1.0,
    "alpha_sigma": 1.0,
}

optim = {
    "name": "adam",
    "lr": 2e-4,
}

train = {
    "training" : True,
    "load_path" : None,
    "run_step" : 100000,
    "print_period" : 1000,
    "save_period" : 10000,
    "test_iteration" : 10,
    # distributed setting
    "distributed_batch_size" : 256,
    "update_period" : 128,
    "num_worker" : 8,
}
