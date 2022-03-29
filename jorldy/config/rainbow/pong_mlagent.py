### Rainbow DQN Pong_ML-Agents Config ###

env = {"name": "pong_mlagent", "time_scale": 12.0}

agent = {
    "name": "rainbow",
    "network": "rainbow",
    "gamma": 0.99,
    "buffer_size": 50000,
    "batch_size": 32,
    "start_train_step": 25000,
    "target_update_period": 1000,
    "lr_decay": True,
    # MultiStep
    "n_step": 3,
    # PER
    "alpha": 0.6,
    "beta": 0.4,
    "learn_period": 4,
    "uniform_sample_prob": 1e-3,
    # Noisy
    "noise_type": "factorized",  # [independent, factorized]
    # C51
    "v_min": -10,
    "v_max": 10,
    "num_support": 51,
}

optim = {
    "name": "adam",
    "lr": 2.5e-4 / 4,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 200000,
    "print_period": 5000,
    "save_period": 50000,
    "eval_iteration": 10,
    # distributed setting
    "update_period": 8,
    "num_workers": 16,
}
