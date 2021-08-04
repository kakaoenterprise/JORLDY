import sys, inspect, re

from torch.optim import *

class_dict = {}
for class_name, _class in inspect.getmembers(sys.modules[__name__], inspect.isclass):
    naming_rule = lambda x: re.sub('([a-z])([A-Z])', r'\1_\2', x).lower()
    class_dict[naming_rule(class_name)] = _class

class Optimizer:  
    def __new__(self, name, *args, **kwargs):
        expected_type = str
        if type(name) != expected_type:
            print("### name variable must be string! ###")
            raise Exception
        name = name.lower()
        if not name in class_dict.keys():
            print(f"### can use only follows {[opt for opt in class_dict.keys()]}")
            raise Exception
        return class_dict[name](*args, **kwargs)
