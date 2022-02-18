### DDPG CartPole Config ###

env = {
    "name": "cartpole",
    "action_type": "continuous",
    "render": False,
}

agent = {
    "name": "td3",
    "actor": "ddpg_actor",
    "critic": "ddpg_critic",
    "gamma": 0.99,
    "buffer_size": 50000,
    "batch_size": 128,
    "start_train_step": 2000,
    "tau": 1e-3,
    "actor_period": 2,
    "act_noise_std": 0.1,
    "target_noise_std": 0.2,
    "target_noise_clip": 0.5,
}

optim = {
    "actor": "adam",
    "critic": "adam",
    "actor_lr": 5e-4,
    "critic_lr": 1e-3,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 100000,
    "print_period": 1000,
    "save_period": 10000,
    "eval_iteration": 10,
    # distributed setting
    "update_period": 1,
    "num_workers": 8,
}
