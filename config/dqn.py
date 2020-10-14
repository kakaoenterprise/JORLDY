from core import *

### Cartpole / DQN ###

# Config
env_name = "cartpole"
mode = "discrete"
agent_name = "dqn"
network = "dqn"
optimizer = "adam"
learning_rate = 1e-4
train_episode = 1000
gamma = 0.99
epsilon_init = 1.0
epsilon_min = 0.1
explore_episode = 0.9
buffer_size = 30000
batch_size = 64
start_train_step = 2000
target_update_term = 1000

env = Env(env_name, mode=mode)
agent = Agent(agent_name,
              state_size=env.state_size,
              action_size=env.action_size,
              network=network,
              optimizer=optimizer,
              learning_rate=learning_rate,
              gamma=gamma,
              epsilon_init=epsilon_init,
              epsilon_min=epsilon_min,
              train_episode=train_episode,
              explore_episode = explore_episode,
              buffer_size=buffer_size,
              batch_size=batch_size,
              start_train_step=start_train_step,
              target_update_term=target_update_term,
              )
