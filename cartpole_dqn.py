from core import *

# Config
cfg = {
"env" : "cartpole",
"network" : "dqn",
"agent" : "dqn",
"optimizer" : "adam",
"learning_rate" : 3e-4,
"episode" : 500,
"gamma" : 0.99,
"epsilon_init" : 0.8,
"epsilon_min" : 0.001,
"epsilon_decay" : 0.001,
"buffer_size" : 50000,
"batch_size" : 64,
"start_train" : 2000,
"update_term" : 500,
}

env = Env(cfg["env"])
network = Network(cfg["network"], env.state_size, env.action_size)
target_network = Network(cfg["network"], env.state_size, env.action_size)
optimizer = Optimizer(cfg["optimizer"], network.parameters(), lr=cfg["learning_rate"])
agent = Agent(cfg["agent"], network=network,
                            target_network=target_network,
                            optimizer=optimizer,
                            gamma=cfg["gamma"],
                            epsilon_init=cfg["epsilon_init"],
                            epsilon_min=cfg["epsilon_min"],
                            epsilon_decay=cfg["epsilon_decay"],
                            buffer_size=cfg["buffer_size"],
                            batch_size=cfg["batch_size"],
                            start_train=cfg["start_train"],
                            update_term=cfg["update_term"],
                            )

for episode in range(cfg["episode"]):
    losses = []
    done = False
    state = env.reset()
    while not done:
        action = agent.act([state])
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        loss = agent.learn()
        losses.append(loss)

    print(f"{episode} Episode / Score : {env.score} / Loss : {sum(losses)/len(losses):.4f} / Epsilon : {agent.epsilon:.4f}")

env.close()