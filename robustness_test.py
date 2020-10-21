from core import *
from utils import Manager
# import config.YOUR_AGENT.YOUR_ENV as config
import config.dqn.cartpole as config

def train(env, agent, id):
    episode = 0
    train_step = 100000
    test_step = 10000
    test_scores = []
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
            if episode%50==0:
                print(f"Process{id} / {episode} Episode / Score : {env.score} / Step : {step} / {manager.get_statistics()}")
            if not training:
                test_scores.append(env.score)
            state = env.reset()
    print(f"### Process{id} / Mean Test Score: {sum(test_scores)/len(test_scores)} ###")
    env.close()
    
if __name__=="__main__":
    from multiprocessing import Process
    import os
    
    mulp = []
    for id in range(os.cpu_count()//4):
        env = Env(name="cartpole")
        agent = Agent(state_size=env.state_size,
                     action_size=env.action_size,
                     **config.agent)
        proc = Process(target=train, args=(env, agent, id))
        mulp.append(proc)
    
    for proc in mulp:
        proc.start()
        
    for proc in mulp:
        proc.join()
    

    