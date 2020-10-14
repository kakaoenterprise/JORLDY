from utils import Manager
import config.dqn as config 

env = config.env
agent = config.agent
train_episode = config.train_episode

step = 0
test_episode = 100
training = True
manager = Manager()

for e in range(train_episode + test_episode):
    if e == train_episode:
        print("### TEST START ###")
        training = False
    done = False
    state = env.reset()
    while not done:
        step += 1
        action = agent.act([state], training)
        next_state, reward, done = env.step(action)
        if training:
            agent.observe(state, action, reward, next_state, done)
            result = agent.learn()
            if result:
                manager.append(result)
        state = next_state

    print(f"{e} Episode / Score : {env.score} / Step : {step} / {manager.get_statistics()}")

env.close()
