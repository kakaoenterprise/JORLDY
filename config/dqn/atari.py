### DQN Config ###

env ={
    "name": "pong",
    "gray_img": True,
    "img_width": 84,
    "img_height": 84,
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
    "explore_step": 8900000,
    "buffer_size": 100000,
    "batch_size": 32,
    "start_train_step": 100000,
    "target_update_term": 10000,
    "print_episode": 5,
}
