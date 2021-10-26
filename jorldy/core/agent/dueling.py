from .dqn import DQN

class Dueling(DQN):
    def __init__(self, *args, **kwargs):
        super(Dueling, self).__init__(*args, **kwargs)