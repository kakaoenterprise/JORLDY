import os
from functools import reduce

import ray

class DistributedManager:
    def __init__(self, Env, env_config, Agent, agent_config, num_worker):
        ray.init()
        agent = Agent(**agent_config)
        num_worker = num_worker if num_worker else os.cpu_count()
        Env, env_config, agent = map(ray.put, [Env, dict(env_config), agent])
        self.actors = [Actor.remote(Env, env_config, agent, i) for i in range(num_worker)]

    def run(self, step=1):
        assert step > 0
        transitions = reduce(lambda x,y: x+y, 
                             ray.get([actor.run.remote(step) for actor in self.actors]))
        return transitions

    def sync(self, sync_item):
        sync_item = ray.put(sync_item)
        ray.get([actor.sync.remote(sync_item) for actor in self.actors])
        
    def terminate(self):
        ray.shutdown()

@ray.remote
class Actor:
    def __init__(self, Env, env_config, agent, id):
        self.env = Env(id=id+1, **env_config)
        self.agent = agent.set_distributed(id)
        self.state = self.env.reset()
    
    def run(self, step):
        transitions = []
        for t in range(step):
            action = self.agent.act(self.state, training=True)
            next_state, reward, done = self.env.step(action)
            transitions.append((self.state, action, reward, next_state, done))
            self.state = next_state if not done else self.env.reset()
        return transitions
    
    def sync(self, sync_item):
        self.agent.sync_in(**sync_item)