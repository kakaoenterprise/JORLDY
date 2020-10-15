from core import *

### DQN ###

# Config
env_name = "cartpole"
mode = "discrete"
agent_name = "dqn"
network = "dqn"
optimizer = "adam"
learning_rate = 5e-5
train_step = 100000
gamma = 0.99
epsilon_init = 1.0
epsilon_min = 0.1
explore_percent = 0.9
buffer_size = 10000
batch_size = 64
start_train_step = 2000
target_update_term = 200

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
              train_step=train_step,
              explore_percent = explore_percent,
              buffer_size=buffer_size,
              batch_size=batch_size,
              start_train_step=start_train_step,
              target_update_term=target_update_term,
              )
