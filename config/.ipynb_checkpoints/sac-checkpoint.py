from core import *

### Cartpole / SAC ###

# Config
env_name = "pendulum"
mode = "continuous"
agent_name = "sac"
actor = "sac_actor"
critic = "sac_critic"
actor_optimizer = "adam"
critic_optimizer = "adam"
alpha_optimizer = "adam"
actor_lr = 5e-4
critic_lr = 1e-3
alpha_lr = 3e-4
train_episode = 500
use_dynamic_alpha = False
gamma = 0.99
tau = 5e-3
buffer_size = 50000
batch_size = 64
start_train_step = 2000
static_log_alpha = -2.0

env = Env(env_name, mode=mode)
agent = Agent(agent_name,
            state_size=env.state_size,
            action_size=env.action_size,
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            alpha_optimizer=alpha_optimizer,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            use_dynamic_alpha=use_dynamic_alpha,
            gamma=gamma,
            tau=tau,
            buffer_size=buffer_size,
            batch_size=batch_size,
            start_train_step=start_train_step,
            static_log_alpha=static_log_alpha,
            )