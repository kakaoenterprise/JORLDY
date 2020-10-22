from core import *
from managers import *

# import config.YOUR_AGENT.YOUR_ENV as config
import config.dqn.atari as config

env = Env(**config.env)
agent = Agent(state_size=env.state_size,
              action_size=env.action_size,
              **config.agent)

episode = 0
train_step = 1000000
test_step = 50000
print_episode = 5
training = True

metric_manager = MetricManager()
board_manager = BoardManager(config.env["name"], config.agent["name"])

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
            metric_manager.append(result)
    state = next_state
    
    if done:
        episode += 1
        metric_manager.append({"score": env.score})
        state = env.reset()
        
        if episode % print_episode == 0:
            statistics = metric_manager.get_statistics()
            print(f"{episode} Episode / Step : {step} / {statistics}")
            board_manager.write_scalar(statistics, step)

env.close()