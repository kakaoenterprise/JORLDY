import numpy as np

# OU noise class 
class OU_Noise:
    def __init__(self, action_size, mu, theta, sigma):
        self.action_size = action_size
        
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        
        self.reset()

    def reset(self):
        self.X = np.ones((1, self.action_size), dtype=np.float32) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X) + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X

# d: dictionary of pytorch tensors which you want to check if they are INF of NaN
# vs: dictionary of pytorch tensors, min and max of which you want to check, when any value of d is INF or NaN
def check_explode(d, vs={}):
    explode = False
    for k in d:
        isnan = d[k].isnan().any()
        isinf = d[k].isinf().any()
        if isnan or isinf:
            print(f"\n\n$$$$$")
            if isnan: print(f"{k} is NaN!!")
            if isinf: print(f"{k} is INF!!")

            print(f"$$$$$\n\n")
            explode = True
    if explode: 
        print("\n\n#####")
        for kv in vs:
            print(f"    {kv}: min {vs[kv].min()} max {vs[kv].max()}")
        print("#####\n\n")
        raise ValueError