from collections import deque

from .per_buffer import PERBuffer
from .multistep_buffer import MultistepBuffer

# Reference: https://github.com/LeejwUniverse/following_deepmid/tree/master/jungwoolee_pytorch/100%20Algorithm_For_RL/01%20sum_tree
class RainbowBuffer(PERBuffer, MultistepBuffer):
    def __init__(self, buffer_size, n_step, num_workers=1, uniform_sample_prob=1e-3):
        MultistepBuffer.__init__(self, buffer_size, n_step, num_workers)
        PERBuffer.__init__(self, buffer_size, uniform_sample_prob)
        
    def store(self, transitions):
        if self.first_store:
            self.check_dim(transitions[0])
        
        assert len(transitions) % self.num_workers == 0
        partition = len(transitions) // self.num_workers
        
        for i, transition in enumerate(transitions):
            # MultiStep
            nstep_buffer = self.nstep_buffers[i//partition]
            nstep_buffer.append(transition)
            if len(nstep_buffer) == self.n_step:
                self.buffer[self.buffer_index] = self.prepare_nstep(nstep_buffer)
        
                # PER
                self.add_tree_data()

                self.buffer_counter = min(self.buffer_counter + 1, self.buffer_size)
                self.buffer_index = (self.buffer_index + 1) % self.buffer_size