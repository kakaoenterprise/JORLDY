import torch
from core import *

# Config
cfg = {
"env" : "cartpole",
"actor" : "sac_actor",
"critic" : "sac_critic",
"agent" : "sac",
"actor_optimizer" : "adam",
"critic_optimizer" : "adam",
"alpha_optimizer" : "adam",
"actor_learning_rate" : 5e-4,
"critic_learning_rate" : 1e-3,
"alpha_learning_rate" : 3e-4,
"episode" : 500,
"use_dynamic_alpha" : False,
"gamma" : 0.99,
"tau" : 5e-3,
"buffer_size" : 50000,
"batch_size" : 64,
"start_train" : 2000,
"static_log_alpha" : -2.0,
}

env = Env(cfg["env"], mode='continuous')
actor = Network(cfg["actor"],env.state_size, env.action_size)
critic = Network(cfg["critic"],env.state_size+env.action_size, env.action_size)
target_critic = Network(cfg["critic"],env.state_size+env.action_size, env.action_size)
actor_optimizer = Optimizer(cfg["actor_optimizer"], actor.parameters(), lr=cfg["actor_learning_rate"])
critic_optimizer = Optimizer(cfg["critic_optimizer"], critic.parameters(), lr=cfg["critic_learning_rate"])

if cfg["use_dynamic_alpha"]:
    log_alpha = torch.zeros(1, requires_grad=True)
    alpha_optimizer = Optimizer(cfg["alpha_optimizer"], [log_alpha], lr=cfg["alpha_learning_rate"])
else:
    log_alpha = None
    alpha_optimizer = None

agent = Agent(cfg["agent"], actor=actor,
                            critic=critic,
                            target_critic=target_critic,
                            actor_optimizer=actor_optimizer,
                            critic_optimizer=critic_optimizer,
                            use_dynamic_alpha=False,
                            log_alpha=log_alpha,
                            alpha_optimizer=alpha_optimizer,
                            gamma=cfg["gamma"],
                            tau=cfg["tau"],
                            buffer_size=cfg["buffer_size"],
                            batch_size=cfg["batch_size"],
                            start_train=cfg["start_train"],
                            static_log_alpha=cfg["static_log_alpha"],
                            )

for episode in range(cfg["episode"]):
    critic_losses1, critic_losses2, actor_losses, alpha_losses = [], [], [], []
    done = False
    state = env.reset()
    while not done:
        action = agent.act([state])
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        results = agent.learn()

    print(f"{episode} Episode / Score : {env.score} /")