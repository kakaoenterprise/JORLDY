import os, sys, inspect, re

file_list = os.listdir(__name__.replace(".","/"))
module_list = [file.replace(".py", "") for file in file_list 
               if file.endswith(".py") and file.replace(".py","") not in ["__init__", "base", "utils"]]
class_dict = {}
for module in module_list:
    module_path = f"{__name__}.{module}"
    __import__(module_path, fromlist=[None])
    for class_name, _class in inspect.getmembers(sys.modules[module_path], inspect.isclass):
        naming_rule = lambda x: re.sub('([a-z])([A-Z])', r'\1_\2', x).lower()
        class_dict[naming_rule(class_name)] = _class

class Network:
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
