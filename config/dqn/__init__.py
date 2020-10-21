### DQN Config ###

agent = {
    "name": "dqn",
    "network": "dqn",
    "optimizer": "adam",
    "learning_rate": 5e-5,
    "gamma": 0.99,
    "epsilon_init": 0.5,
    "epsilon_min": 0.05,
    "explore_step": 90000,
    "buffer_size": 50000,
    "batch_size": 64,
    "start_train_step": 5000,
    "target_update_term": 100,
}
