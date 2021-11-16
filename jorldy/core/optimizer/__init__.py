import os, sys, inspect, re
from collections import OrderedDict

from torch.optim import *

optimizer_dict = {}
naming_rule = lambda x: re.sub("([a-z])([A-Z])", r"\1_\2", x).lower()
for class_name, _class in inspect.getmembers(sys.modules[__name__], inspect.isclass):
    optimizer_dict[naming_rule(class_name)] = _class

working_path = os.path.dirname(os.path.realpath(__file__))
optimizer_dict = OrderedDict(sorted(optimizer_dict.items()))
with open(os.path.join(working_path, "_optimizer_dict.txt"), "w") as f:
    f.write("### Optimizer Dictionary ###\n")
    f.write("format: (key, class)\n")
    f.write("------------------------\n")
    for item in optimizer_dict.items():
        f.write(str(item) + "\n")


class Optimizer:
    def __new__(self, name, *args, **kwargs):
        expected_type = str
        if type(name) != expected_type:
            print("### name variable must be string! ###")
            raise Exception
        name = name.lower()
        if not name in optimizer_dict.keys():
            print(f"### can use only follows {[opt for opt in optimizer_dict.keys()]}")
            raise Exception
        return optimizer_dict[name](*args, **kwargs)
