### MPO Procgen Config ###

env = {
    # "name": it should be defined in the command. ex) python main.py --config config.AGENT.procgen --env.name coinrun
    "render": False,
    "gray_img": True,
    "stack_frame": 4,
    "no_op": True,
    "skip_frame": 4,
    "reward_clip": True,
}

agent = {
    "name": "mpo",
    "actor": "discrete_policy",
    "head": "cnn",
    "critic": "discrete_q_network",
    "critic_loss_type": "retrace",  # one of ['1step_TD', 'retrace']
    "gamma": 0.99,
    "buffer_size": 50000,
    "batch_size": 64,
    "n_step": 8,
    "start_train_step": 2000,
    "n_epoch": 64,  # not tested (20210830)
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
    "lr": 2.5e-4,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 30000000,
    "print_period": 10000,
    "save_period": 100000,
    "eval_iteration": 5,
    "record": True,
    "record_period": 300000,
    # distributed setting
    "distributed_batch_size": 256,
    "update_period": 128,
    "num_workers": 8,
}
