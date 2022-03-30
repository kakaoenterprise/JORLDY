### Ape-X Super Mario Bros Config ###

env = {
    "name": "super_mario_bros",
    "render": False,
    "gray_img": True,
    "img_width": 84,
    "img_height": 84,
    "stack_frame": 4,
}

agent = {
    "name": "ape_x",
    "network": "dueling",
    "head": "cnn",
    "gamma": 0.99,
    "buffer_size": 2000000,
    "batch_size": 32,
    "clip_grad_norm": 40.0,
    "start_train_step": 50000,
    "target_update_period": 2500,
    "lr_decay": True,
    # MultiStep
    "n_step": 3,
    # PER
    "alpha": 0.6,
    "beta": 0.4,
    "uniform_sample_prob": 1e-3,
}

optim = {
    "name": "rmsprop",
    "eps": 1.5e-7,
    "lr": 2.5e-4 / 4,
    "centered": True,
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
    "distributed_batch_size": 512,
    "update_period": 100,
    "num_workers": 128,
}
