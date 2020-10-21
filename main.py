from core import *
from utils import LogManager
import time 

# import config.YOUR_AGENT.YOUR_ENV as config
import config.dqn.atari as config

env = Env(**config.env)
agent = Agent(state_size=env.state_size,
              action_size=env.action_size,
              **config.agent)

episode = 0
train_step = 100000
test_step = 10000
training = True
log_manager = LogManager()

state = env.reset()
for step in range(train_step + test_step):
    if step == train_step:
        print("### TEST START ###")
        training = False
    
    action = agent.act([state], training)
    next_state, reward, done = env.step(action)
    
    if training:
        agent.observe(state, action, reward, next_state, done)
        result = agent.learn()
        if result:
            log_manager.append(result)
    state = next_state
    
    if done:
        episode += 1
        print(f"{episode} Episode / Score : {env.score} / Step : {step} / {log_manager.get_statistics()}")
        state = env.reset()

env.close()