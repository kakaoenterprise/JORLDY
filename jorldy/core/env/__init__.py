import os, sys, inspect, re, traceback
from collections import OrderedDict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # for import mlagents

working_path = os.path.dirname(os.path.realpath(__file__))
file_list = os.listdir(working_path)
module_list = [
    file.replace(".py", "")
    for file in file_list
    if file.endswith(".py")
    and file.replace(".py", "") not in ["__init__", "base", "utils"]
]
env_dict = {}
error_dict = {}
naming_rule = lambda x: re.sub("([a-z])([A-Z])", r"\1_\2", x).lower()
all_modules_available = True
for module_name in module_list:
    module_path = f"{__name__}.{module_name}"
    try:
        module = __import__(module_path, fromlist=[None])
    except Exception:
        all_modules_available = False
        error_dict[module_name] = traceback.format_exc()
        continue
    else:
        for class_name, _class in inspect.getmembers(module, inspect.isclass):
            if module_path in str(_class) and "_" != class_name[0]:
                env_dict[naming_rule(class_name)] = _class

if all_modules_available:
    env_dict = OrderedDict(sorted(env_dict.items()))
    with open(os.path.join(working_path, "_env_dict.txt"), "w") as f:
        f.write("### Env Dictionary ###\n")
        f.write("format: (key, class)\n")
        f.write("------------------------\n")
        for item in env_dict.items():
            f.write(str(item) + "\n")


class Env:
    def __new__(self, name, *args, **kwargs):
        expected_type = str
        if type(name) != expected_type:
            print("### name variable must be string! ###")
            raise Exception
        name = name.lower()
        if not name in env_dict.keys():
            print(f"### can use only follows {[opt for opt in env_dict.keys()]}")
            print(
                "============================================================================"
            )
            print(
                "If you try to use the following modules, please refer to the error contents."
            )
            print(f"Unavailable moduels {list(error_dict)}")
            print(
                "============================================================================"
            )
            for module, error in error_dict.items():
                print(f"module: {module}")
                print(f"error: {error}")
            raise Exception
        return env_dict[name](*args, **kwargs)
