from .dqn import DQNAgent

class DuelingAgent(DQNAgent):
    def __init__(self, *args, **kwargs):
        super(DuelingAgent, self).__init__(*args, **kwargs)