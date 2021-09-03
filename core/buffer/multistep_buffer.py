from collections import deque
import numpy as np

from .replay_buffer import ReplayBuffer

class MultistepBuffer(ReplayBuffer):
    def __init__(self, buffer_size, n_step, num_workers):
        super(MultistepBuffer, self).__init__(buffer_size)
        self.n_step = n_step
        self.num_workers = num_workers
        self.nstep_buffers = [deque(maxlen=self.n_step) for _ in range(num_workers)]
        
    def prepare_nstep(self, batch):
        transition = {}
        transition['state'] = batch[0]['state']
        transition['next_state'] = batch[-1]['next_state']
        
        for key in batch[0].keys():
            if key not in ['state', 'next_state']:
                transition[key] = np.stack([b[key] for b in batch], axis=1)
        
        return transition
        
    def store(self, transitions):
        if self.first_store:
            self.check_dim(transitions[0])

        assert len(transitions) % self.num_workers == 0
        partition = len(transitions) // self.num_workers
        
        for i, transition in enumerate(transitions):
            nstep_buffer = self.nstep_buffers[i//partition]
            nstep_buffer.append(transition)
            if len(nstep_buffer) == self.n_step:
                self.buffer.append(self.prepare_nstep(nstep_buffer))