import ray
import time

ray.init()

@ray.remote
class Counter(object):
    def __init__(self, i):
        self.n = i

    def increment(self):
        self.n += 1

    def read(self):
        return self.n

counters = [Counter.remote(i) for i in range(4)]
[c.increment.remote() for c in counters]
futures = [c.read.remote() for c in counters]
print(ray.get(futures))
ray.timeline()

