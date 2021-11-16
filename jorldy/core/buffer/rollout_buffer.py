import numpy as np

from .base import BaseBuffer


class RolloutBuffer(BaseBuffer):
    def __init__(self):
        super(RolloutBuffer, self).__init__()
        self.buffer = list()

    def store(self, transitions):
        if self.first_store:
            self.check_dim(transitions[0])
        self.buffer += transitions

    def sample(self):
        transitions = self.stack_transition(self.buffer)

        self.buffer.clear()
        return transitions

    @property
    def size(self):
        return len(self.buffer)
