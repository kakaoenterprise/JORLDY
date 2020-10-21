### DQN Config ###

agent = {
    "name": "dqn",
    "network": "dqn",
    "optimizer": "adam",
    "learning_rate": 5e-4,
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "epsilon_min": 0.1,
    "explore_step": 80000,
    "buffer_size": 50000,
    "batch_size": 64,
    "start_train_step": 10000,
    "target_update_term": 100,
}
