from core import *
from managers import *

# import config.YOUR_AGENT.YOUR_ENV as config
import config.dqn.cartpole as config

if __name__=="__main__":
    env = Env(**config.env)
    agent = Agent(state_size=env.state_size,
                  action_size=env.action_size,
                  **config.agent)

    load_path = config.train["load_path"]
    assert load_path is not None
    if load_path:
        agent.load(load_path)

    run_step = config.train["run_step"]

    episode, score = 0, 0
    state = env.reset()
    for step in range(1, run_step+1):
        action = agent.act(state, training)
        next_state, reward, done = env.step(action)
        score += reward
        state = next_state
        if done:
            episode += 1
            state = env.reset()
            print(f"{episode} Episode / Step : {step} / Score: {score}")
            score = 0

    env.close()
