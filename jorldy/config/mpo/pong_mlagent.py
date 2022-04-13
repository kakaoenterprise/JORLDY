### MPO Pong_ML-Agents Config ###

env = {"name": "pong_mlagent", "time_scale": 12.0}

agent = {
    "name": "mpo",
    "actor": "discrete_policy",
    "critic": "discrete_q_network",
    "critic_loss_type": "1step_TD",  # one of ['1step_TD', 'retrace']
    "gamma": 0.99,
    "buffer_size": 50000,
    "batch_size": 64,
    "n_step": 1,
    "start_train_step": 2000,
    "n_epoch": 64,
    "clip_grad_norm": 1.0,
    "min_eta": 1e-8,
    "min_alpha_mu": 1e-8,
    "min_alpha_sigma": 1e-8,
    "eps_eta": 0.01,
    "eps_alpha_mu": 0.01,
    "eps_alpha_sigma": 5 * 1e-5,
    "eta": 1.0,
    "alpha_mu": 1.0,
    "alpha_sigma": 1.0,
    "lr_decay": True,
}

optim = {
    "name": "adam",
    "lr": 5e-4,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 200000,
    "print_period": 1000,
    "save_period": 10000,
    "eval_iteration": 10,
    # distributed setting
    "distributed_batch_size": 256,
    "update_period": 128,
    "num_workers": 16,
}
