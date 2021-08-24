import os, sys, inspect, re
from collections import OrderedDict

from torch.optim import *

class_dict = OrderedDict()
for class_name, _class in inspect.getmembers(sys.modules[__name__], inspect.isclass):
    naming_rule = lambda x: re.sub('([a-z])([A-Z])', r'\1_\2', x).lower()
    class_dict[naming_rule(class_name)] = _class

working_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(working_path, "_class_dict.txt"), 'w') as f:
    f.write('### Class Dictionary ###\n')
    f.write('format: (key, class)\n')
    f.write('------------------------\n')
    for item in class_dict.items():
        f.write(str(item) + '\n')
    
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
