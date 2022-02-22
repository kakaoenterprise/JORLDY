### TD3 Mujoco Config ###

env = {
    # "name": it should be defined in the command. ex) python main.py --config config.AGENT.mujoco --env.name half_cheetah
    "render": False,
}
agent = {
    "name": "td3",
    "actor": "ddpg_actor",
    "critic": "ddpg_critic",
    "gamma": 0.99,
    "buffer_size": 1000000,
    "batch_size": 128,
    "start_train_step": 1000,
    "initial_random_step": 1000,
    "tau": 5e-3,
    "update_delay": 2,
    "action_noise_std": 0.1,
    "target_noise_std": 0.2,
    "target_noise_clip": 0.5,
}

optim = {
    "actor": "adam",
    "critic": "adam",
    "actor_lr": 1e-3,
    "critic_lr": 1e-3,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 1000000,
    "print_period": 5000,
    "save_period": 10000,
    "eval_iteration": 10,
    # distributed setting
    "distributed_batch_size": 256,
    "update_period": 1,
    "num_workers": 8,
}
