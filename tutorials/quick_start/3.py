import ray
import time

ray.init()

@ray.remote
class Actor(object):
    def __init__(self, i):
        self.n = i

    def append(self, l):
        l.append(self.n)
        print(l)

l = list()
l = ray.put(l)
actors = [Actor.remote(i) for i in range(4)]
[a.append.remote(l) for a in actors]
print(ray.get(l))

