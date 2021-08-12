import argparse

from core import *
from manager import *

from collections import deque
import matplotlib.pyplot as plt 
import numpy as np
import imageio
import cv2
import torch

# default_config_path = "config.YOUR_AGENT.YOUR_ENV"
# default_config_path = "config.icm_ppo.mario"
# default_config_path = "config.rnd_ppo.mario-debug"
default_config_path = "config.rnd_ppo.montezuma_debug"

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
    frames = []
    r_is = deque(maxlen=20)
    score = 0
    
    for step in range(1, config.train.run_step+1):
        action = agent.act(state, training=True)            
        if step < 20:
            action = np.random.randint(0, 5, size=(1, 1))

        next_state, reward, done = env.step(action)
        frames.append(env.get_frame())
        r_i = agent.rnd.forward(torch.as_tensor(next_state, dtype=torch.float32, device=agent.device))
        r_is.append(int(r_i))
        cv2.putText(frames[-1], f"{r_i.item():.1f}", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for i in range(min(step, 20)):
            cv2.putText(frames[-1], f".", (50+i*2,120-r_is[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        x_plot.append(step)
        y_plot.append(r_i.item())
        if done:
            plt.rcParams["figure.figsize"] = (18,6)
            plt.scatter(x_plot, y_plot, s=0.5)
            plt.savefig(f"./{step:010d}_{env.score}.png")

            imageio.mimwrite(f"./{step:010d}_{env.score}.gif", frames, fps=30)
            print('Plot is done')
            break
        
        state = next_state
        
    env.close()
'''
    def get_ri(self, state, next_state, action):
        state, next_state, action = map(lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device), [state, next_state, action])
        r_i, _, _ = self.icm(state, action, next_state)
        return r_i
'''