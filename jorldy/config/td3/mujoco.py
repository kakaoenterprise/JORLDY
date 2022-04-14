### TD3 Mujoco Config ###

env = {
    # "name": it should be defined in the command. ex) python main.py --config config.AGENT.mujoco --env.name half_cheetah
    "render": False,
}
agent = {
    "name": "td3",
    "actor": "deterministic_policy",
    "critic": "continuous_q_network",
    "hidden_size": 512,
    "gamma": 0.99,
    "buffer_size": 1000000,
    "batch_size": 128,
    "start_train_step": 25000,
    "initial_random_step": 25000,
    "tau": 5e-3,
    "update_delay": 2,
    "action_noise_std": 0.1,
    "target_noise_std": 0.2,
    "target_noise_clip": 0.5,
    "lr_decay": True,
}

optim = {
    "actor": "adam",
    "critic": "adam",
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 1000000,
    "print_period": 10000,
    "save_period": 100000,
    "eval_iteration": 10,
    # distributed setting
    "distributed_batch_size": 256,
    "update_period": 1,
    "num_workers": 8,
}
