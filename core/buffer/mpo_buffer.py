from collections import deque
import numpy as np

from .multistep_buffer import MultistepBuffer

class MPOBuffer(MultistepBuffer):
    def __init__(self, buffer_size, n_step, num_workers):
        super(MPOBuffer, self).__init__(buffer_size, n_step, num_workers)
        
    def prepare_nstep(self, batch):
        transition = {}
        
        for key in batch[0].keys():
            transition[key] = np.stack([b[key] for b in batch], axis=1)
        
        return transition