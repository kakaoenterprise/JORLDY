import os, sys, inspect, re

sys.path.append(os.path.abspath('../../'))

working_path = __name__.replace(".","/")
file_list = os.listdir(working_path)
module_list = [file.replace(".py", "") for file in file_list 
               if file.endswith(".py") and file.replace(".py","") not in ["__init__", "base", "utils"]]
class_dict = {}
for module in module_list:
    module_path = f"{__name__}.{module}"
    __import__(module_path, fromlist=[None])
    for class_name, _class in inspect.getmembers(sys.modules[module_path], inspect.isclass):
        if module_path in str(_class):
            naming_rule = lambda x: re.sub('([a-z])([A-Z])', r'\1_\2', x.replace('Agent','')).lower()
            class_dict[naming_rule(class_name)] = _class

with open(os.path.join(working_path, "_class_dict.txt"), 'w') as f:
    f.write('### Class Dictionary ###\n')
    f.write('format: (key, class)\n')
    f.write('------------------------\n')
    for item in class_dict.items():
        f.write(str(item) + '\n')
        
class Agent:
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
