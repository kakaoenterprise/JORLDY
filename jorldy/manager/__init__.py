import os, inspect

working_path = os.path.dirname(os.path.realpath(__file__))
file_list = os.listdir(working_path)
module_list = [
    file.replace(".py", "")
    for file in file_list
    if file.endswith(".py") and file.replace(".py", "") not in ["__init__"]
]

for module_name in module_list:
    module_path = f"{__name__}.{module_name}"
    module = __import__(module_path, fromlist=[None])
    for class_name, _class in inspect.getmembers(module, inspect.isclass):
        if module_path in str(_class) and "Manager" in class_name:
            exec(f"from {module_path} import {class_name}")
