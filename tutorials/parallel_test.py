import ray
import time
import copy


class Counter:
    def run(self):
        time.sleep(1.0)

@ray.remote
class Actor:
    def __init__(self, counter):
        self.counter = counter
    
    def run(self, step):
        for _ in range(step):
            self.counter.run()
        
class Manager:
    def __init__(self, counter):
        ray.init()
        counter = copy.deepcopy(counter)
        self.actors = [Actor.remote(counter) for _ in range(16)]

    def run(self):
        ray.get([actor.run.remote(4) for actor in self.actors])
        
        
if __name__=="__main__":
    counter = Counter()
    manager = Manager(counter)
    
    t = time.time()
    manager.run()
    print("took time:", time.time() - t)