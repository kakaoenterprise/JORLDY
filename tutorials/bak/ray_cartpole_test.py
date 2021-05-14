import ray
import gym
import time
import numpy as np 
import psutil
import os

for proxy in ['https_proxy', 'http_proxy']:
    if os.environ.get(proxy):
        del os.environ[proxy]

num_agent = 10
action_size = 2
test_step = 100000
replay_memory = []
num_cpu = psutil.cpu_count(logical=False)
print("Number of CPUs: {}".format(num_cpu))

ray.init()

@ray.remote
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

agents = [Agent.remote() for _ in range(num_agent)]
env = gym.make('CartPole-v0')        
env_remote = ray.put(env)
start_time = time.time()
results = [agent.run.remote(env_remote, test_step) for agent in agents]
for res in ray.get(results):
    replay_memory.extend(res)

print(len(replay_memory))
print("Computation time: {}".format(time.time()-start_time))
ray.timeline()
