import argparse

from core import *
from manager import *

# default_config_path = "config.YOUR_AGENT.YOUR_ENV"
default_config_path = "config.dqn.cartpole"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config.dqn.cartpole")
    args, unknown = parser.parse_known_args()
    config_path = args.config if args.config else default_config_path
    config_manager = ConfigManager(config_path, unknown)
    config = config_manager.config

    env = Env(**config.env)
    agent_config = {
        "state_size": env.state_size,
        "action_size": env.action_size,
        "optim_config": config.optim,
    }
    agent_config.update(config.agent)
    agent = Agent(**agent_config)

    assert config.train.load_path
    agent.load(config.train.load_path)

    episode, score = 0, 0
    state = env.reset()
    for step in range(1, config.train.run_step + 1):
        action = agent.act(state, training=False)
        next_state, reward, done = env.step(action)
        score += reward
        state = next_state
        if done:
            episode += 1
            state = env.reset()
            print(f"{episode} Episode / Step : {step} / Score: {score}")
            score = 0

    env.close()
