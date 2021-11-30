from .dqn import DQN


class Dueling(DQN):
    def __init__(self, *args, **kwargs):
        if "network" in kwargs.keys():
            kwargs["network"] = "dueling"
        assert kwargs["network"] == "dueling"
        super(Dueling, self).__init__(*args, **kwargs)
