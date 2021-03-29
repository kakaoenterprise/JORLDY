import gym
import time
import numpy as np 

num_agent = 10
action_size = 2
test_step = 100000
replay_memory = []

class Agent:
    def act(self):
        return np.random.randint(action_size)
    
    def run(self, env, step):
        state = env.reset()

        transitions = []
        for _ in range(step):
            action = self.act()
            next_state, reward, done, info = env.step(action)
            transitions.append((state, action, reward, next_state, done))

            state = env.reset() if done else next_state

        return transitions
    
env = gym.make('CartPole-v0')        
agents = [Agent() for _ in range(num_agent)] 

start_time = time.time()
for i in range(num_agent):
    replay_memory.extend(agents[i].run(env, test_step)) 
    
print(len(replay_memory))
print("Computation time: {}".format(time.time()-start_time))
