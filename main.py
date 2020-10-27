from core import *
from managers import LogManager, BoardManager

# import config.YOUR_AGENT.YOUR_ENV as config
import config.dqn.atari as config

env = Env(**config.env)
agent = Agent(state_size=env.state_size,
              action_size=env.action_size,
              **config.agent)

episode = 0

score_sum = 0
loss_list = []

train_step = 1000000
test_step = 50000
training = True

log_manager = LogManager()
board_manager = BoardManager(config.agent, config.env)

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
        score_sum += env.score
        state = env.reset()
        
        if episode % config.agent['print_episode'] == 0:
            mean_score = score_sum/config.agent['print_episode']
            print(f"{episode} Episode / Score : {mean_score} / Step : {step} / {log_manager.get_statistics()}")
            board_manager.write_scalar(mean_score, step)
            score_sum = 0

env.close()