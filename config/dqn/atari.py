### DQN Config ###

env ={
    "name": "breakout",
    "mode": "discrete",
    "gray_img": True,
    "img_width": 80,
    "img_height": 80,
    "stack_frame": 4,
}

agent = {
    "name": "dqn",
    "network": "dqn_cnn",
    "optimizer": "adam",
    "learning_rate": 0.00025,
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.1,
    "explore_step": 80000,
    "buffer_size": 50000,
    "batch_size": 64,
    "start_train_step": 10000,
    "target_update_term": 1000,
}
