import os, sys, inspect, re
from collections import OrderedDict

sys.path.append(os.path.abspath("../../"))

working_path = os.path.dirname(os.path.realpath(__file__))
file_list = os.listdir(working_path)
module_list = [
    file.replace(".py", "")
    for file in file_list
    if file.endswith(".py")
    and file.replace(".py", "") not in ["__init__", "base", "utils"]
]
agent_dict = {}
naming_rule = lambda x: re.sub("([a-z])([A-Z])", r"\1_\2", x).lower()
for module_name in module_list:
    module_path = f"{__name__}.{module_name}"
    module = __import__(module_path, fromlist=[None])
    for class_name, _class in inspect.getmembers(module, inspect.isclass):
        if module_path in str(_class):
            agent_dict[naming_rule(class_name)] = _class

agent_dict = OrderedDict(sorted(agent_dict.items()))
with open(os.path.join(working_path, "_agent_dict.txt"), "w") as f:
    f.write("### Agent Dictionary ###\n")
    f.write("format: (key, class)\n")
    f.write("------------------------\n")
    for item in agent_dict.items():
        f.write(str(item) + "\n")


class Agent:
    def __new__(self, name, *args, **kwargs):
        expected_type = str
        if type(name) != expected_type:
            print("### name variable must be string! ###")
            raise Exception
        name = name.lower()
        if not name in agent_dict.keys():
            print(f"### can use only follows {[opt for opt in agent_dict.keys()]}")
            raise Exception
        return agent_dict[name](*args, **kwargs)
