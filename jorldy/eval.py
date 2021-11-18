import argparse

from core import *
from manager import *

# default_config_path = "config.YOUR_AGENT.YOUR_ENV"
default_config_path = "config.ppo.breakout"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config.dqn.cartpole")
    args, unknown = parser.parse_known_args()
    config_path = args.config if args.config else default_config_path
    config_manager = ConfigManager(config_path, unknown)
    config = config_manager.config

    env = Env(**config.env)
    agent = Agent(
        state_size=env.state_size, action_size=env.action_size, **config.agent
    )

    assert config.train.load_path
    agent.load(config.train.load_path)

    episode, score = 0, 0
    state = env.reset()
    for step in range(1, config.train.run_step + 1):
        action_dict = agent.act(state, training=False)
        next_state, reward, done = env.step(action_dict["action"])
        transition = {
                    "state": state,
                    "next_state": next_state,
                    "reward": reward,
                    "done": done,
                }
        transition.update(action_dict)
        agent.interact_callback(transition)
        score += reward
        state = next_state
        if done:
            episode += 1
            state = env.reset()
            print(f"{episode} Episode / Step : {step} / Score: {score}")
            score = 0

    env.close()
