import os


class ConfigManager:
    def __init__(self, config_path, unknown_args=[]):
        module = __import__(config_path, fromlist=[None])
        self.config = CustomDict()
        self.config.agent = CustomDict(module.agent)
        self.config.optim = CustomDict(module.optim)
        self.config.env = CustomDict(module.env)
        self.config.train = CustomDict(module.train)

        self.unknown_update(unknown_args)

    def unknown_update(self, unknown_args):
        remove_list = []
        idx = 0
        while idx < len(unknown_args):
            # Get query
            query = unknown_args[idx]
            assert "--" in query, "use -- before the optional argument."

            # Get key and value
            if "=" in query:
                key, value = query.strip("-").split("=")
            else:
                key = query.strip("-")
                idx += 1
                assert (
                    idx < len(unknown_args) and "--" not in unknown_args[idx]
                ), "check command again."
                value = unknown_args[idx]

            # Get domain and key
            assert "." in key and key.split(".")[0] in [
                "env",
                "agent",
                "optim",
                "train",
            ], "optional argument should include env, agent or train. ex)env.name"
            domain, key = key.split(".")

            # Update
            value = type_cast(value)
            if value is None:
                remove_list.append((domain, key))
            else:
                self.config[domain][key] = value

            idx += 1

        for domain, key in remove_list:
            self.config[domain].pop(key, None)

    def dump(self, dump_path):
        tab, newline, open_brace, close_brace = "\t", "\n", "{", "}"
        with open(os.path.join(dump_path, "config.py"), "w", encoding="utf-8") as f:
            f.write(
                f"### {self.config.agent.name} {self.config.env.name} config ###{newline}"
            )

            for domain in self.config.keys():
                f.write(f"{newline}")
                f.write(f"{domain} = {open_brace}{newline}")
                for key, value in self.config[domain].items():
                    value = f"'{value}'" if type(value) == str else value
                    f.write(f"{tab}'{key}': {value},{newline}")
                f.write(f"{close_brace}{newline}")


class CustomDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __getitem__ = __getattr__

    def __init__(self, init_dict={}):
        self.update(init_dict)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)


def type_cast(var):
    try:
        return int(var)
    except:
        try:
            return float(var)
        except:
            try:
                assert var in ["True", "False"]
                return True if var == "True" else False
            except:
                return None if var == "None" else var
