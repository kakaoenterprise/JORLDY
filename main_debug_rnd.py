import argparse

from core import *
from manager import *

import matplotlib.pyplot as plt 
import numpy as np

# default_config_path = "config.YOUR_AGENT.YOUR_ENV"
default_config_path = "config.rnd_dqn.mario"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config.dqn.cartpole')
    args, unknown = parser.parse_known_args()
    config_path = args.config if args.config else default_config_path
    config_manager = ConfigManager(config_path, unknown)
    config = config_manager.config
    
    env = Env(**config.env)
    agent = Agent(state_size=env.state_size,
                  action_size=env.action_size,
                  **config.agent)

    if config.train.load_path:
        agent.load(config.train.load_path)
    
    episode = 0
    state = env.reset()
    
    count_step = 0
    x_plot = []
    y_plot = []
    score = 0
    
    for step in range(1, config.train.run_step+1):
        action = agent.act(state, config.train.training)            

        if step < 20:
            action = np.random.randint(0, 7, size=(1, 1))
        
        next_state, reward, done = env.step(action)
        
        score += reward[0][0]
        
        r_i = agent.get_ri(next_state)
        
        x_plot.append(step)
        y_plot.append(r_i.item())
        
        if done:
            plt.plot(x_plot, y_plot)
            plt.savefig('./r_i (' + str(score) +').png')
            raise Exception('Plot is done')
        
        state = next_state

        if done:
            episode += 1
            state = env.reset()
        
    env.close()
