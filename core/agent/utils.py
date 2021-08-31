import torch
import numpy as np

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