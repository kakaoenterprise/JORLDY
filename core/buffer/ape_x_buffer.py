from collections import deque
import numpy as np

from .rainbow_buffer import RainbowBuffer

class ApeXBuffer(RainbowBuffer):
    def __init__(self, gamma, *args, **kwargs):
        super(ApeXBuffer, self).__init__(*args, **kwargs)
        self.gamma = gamma
        
    def store(self, transitions, delta_t=1):
        if self.first_store:
            self.check_dim(transitions[0])
        
        assert len(transitions) % self.num_worker == 0
        partition = len(transitions) // self.num_worker
        
        for i, transition in enumerate(transitions):
            # MultiStep
            nstep_buffer = self.nstep_buffers[i//partition]
            nstep_buffer.append(transition)
            if len(nstep_buffer) == self.n_step:
                self.buffer[self.buffer_index], priority = self.prepare_nstep(nstep_buffer)
                # PER
                self.add_tree_data(priority)

                self.buffer_counter = min(self.buffer_counter + 1, self.buffer_size)
                self.buffer_index = (self.buffer_index + 1) % self.buffer_size
    
    def prepare_nstep(self, batch):
        transition = {}
        transition['state'] = batch[0]['state']
        transition['next_state'] = batch[-1]['next_state']
        
        priority = batch[0]['priority']

        for key in batch[0].keys():
            if key not in ['state', 'next_state']:
                transition[key] = np.stack([b[key] for b in batch], axis=1)

        return transition, priority