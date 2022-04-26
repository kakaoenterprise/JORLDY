import os
from functools import reduce

import ray


class DistributedManager:
    def __init__(self, Env, env_config, Agent, agent_config, num_workers, mode):
        assert ray.is_initialized() == False
        try:
            ray.init(address="auto")
        except:
            ray.init()
        agent = Agent(**agent_config)
        self.num_workers = num_workers if num_workers else os.cpu_count()
        Env, env_config, agent = map(ray.put, [Env, dict(env_config), agent])
        self.actors = [
            Actor.remote(Env, env_config, agent, i) for i in range(self.num_workers)
        ]

        assert mode in ["sync", "async"]
        self.mode = mode
        self.sync_item = None
        self.running_ids = []

    def run(self, step=1):
        assert step > 0
        if self.mode == "sync":
            items = ray.get([actor.run.remote(step) for actor in self.actors])
            transitions = reduce(lambda x, y: x + y, [item[1] for item in items])
            completed_ratio = 1.0
        else:
            if len(self.running_ids) == 0:
                self.running_ids = [actor.run.remote(step) for actor in self.actors]

            done_ids = []
            while len(done_ids) == 0:
                done_ids, self.running_ids = ray.wait(
                    self.running_ids, num_returns=self.num_workers, timeout=0.1
                )

            items = ray.get(done_ids)
            transitions = reduce(lambda x, y: x + y, [item[1] for item in items])
            runned_ids = [item[0] for item in items]
            completed_ratio = len(items) / self.num_workers

            if self.sync_item is not None:
                ray.get(
                    [self.actors[id].sync.remote(self.sync_item) for id in runned_ids]
                )
            self.running_ids += [self.actors[id].run.remote(step) for id in runned_ids]

        return transitions, completed_ratio

    def sync(self, sync_item, init=False):
        if self.mode == "sync" or init:
            sync_item = ray.put(sync_item)
            ray.get([actor.sync.remote(sync_item) for actor in self.actors])
        else:
            self.sync_item = ray.put(sync_item)

    def terminate(self):
        if len(self.running_ids) > 0:
            ray.get(self.running_ids)
        ray.shutdown()


@ray.remote
class Actor:
    def __init__(self, Env, env_config, agent, id):
        self.id = id
        self.env = Env(id=id + 1, **env_config)
        self.agent = agent.set_distributed(id)
        self.state = self.env.reset()

    def run(self, step):
        transitions = []
        for t in range(step):
            action_dict = self.agent.act(self.state, training=True)
            next_state, reward, done = self.env.step(action_dict["action"])
            transition = {
                "state": self.state,
                "next_state": next_state,
                "reward": reward,
                "done": done,
            }
            transition.update(action_dict)
            transition = self.agent.interact_callback(transition)
            if transition:
                transitions.append(transition)
            self.state = next_state if not done else self.env.reset()
        return self.id, transitions

    def sync(self, sync_item):
        self.agent.sync_in(**sync_item)
