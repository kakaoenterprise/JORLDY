# import config.YOUR_CONFIG as config
import config.dqn as config 

from utils import Manager

env = config.env
agent = config.agent
train_step = config.train_step

episode = 0
test_step = 10000
training = True
manager = Manager()

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
            manager.append(result)
    state = next_state
    
    if done:
        episode += 1
        print(f"{episode} Episode / Score : {env.score} / Step : {step} / {manager.get_statistics()}")
        state = env.reset()

env.close()