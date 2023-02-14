import numpy as np

from .replay_buffer import ReplayBuffer


# Reference: https://github.com/LeejwUniverse/following_deepmid/tree/master/jungwoolee_pytorch/100%20Algorithm_For_RL/01%20sum_tree
class PERBuffer(ReplayBuffer):
    def __init__(self, buffer_size, uniform_sample_prob=1e-3):
        super(PERBuffer, self).__init__(buffer_size)
        self.tree_size = (self.buffer_size * 2) - 1
        self.first_leaf_index = self.buffer_size - 1

        self.sum_tree = np.zeros(self.tree_size)  # define sum tree
        self.tree_index = self.first_leaf_index  # define sum_tree leaf node index.

        self.max_priority = 1.0
        self.uniform_sample_prob = uniform_sample_prob

    def store(self, transitions):
        if self.first_store:
            self.check_dim(transitions[0])

        for transition in transitions:
            self.buffer[self.buffer_index] = transition
            new_priority = (
                transition["priority"]
                if "priority" in transition
                else self.max_priority
            )
            self.add_tree_data(new_priority)

            self.buffer_counter = min(self.buffer_counter + 1, self.buffer_size)
            self.buffer_index = (self.buffer_index + 1) % self.buffer_size

    def add_tree_data(self, new_priority):
        self.update_priority(new_priority, self.tree_index)

        self.tree_index += 1  # count current sum_tree index
        if self.tree_index == self.tree_size:  # if sum tree index achive last index.
            self.tree_index = self.first_leaf_index  # change frist leaf node index.

    def update_priority(self, new_priority, index):
        ex_priority = self.sum_tree[index]
        delta_priority = new_priority - ex_priority
        self.sum_tree[index] = new_priority
        self.update_tree(index, delta_priority)

        self.max_priority = max(self.max_priority, new_priority)

    def update_tree(self, index, delta_priority):
        # index is a starting leaf node point.
        while index > 0:
            index = (index - 1) // 2  # parent node index.
            self.sum_tree[index] += delta_priority

    def search_tree(self, num):
        index = 0  # always start from root index.
        while index < self.first_leaf_index:
            left = (index * 2) + 1
            right = (index * 2) + 2

            if num <= self.sum_tree[left]:  # if child left node is over current value.
                index = left  # go to the left direction.
            else:
                num -= self.sum_tree[left]  # if child left node is under current value.
                index = right  # go to the right direction.

        return index

    def sample(self, beta, batch_size):
        assert self.sum_tree[0] > 0.0
        uniform_sampling = np.random.uniform(size=batch_size) < self.uniform_sample_prob
        uniform_size = np.sum(uniform_sampling)
        prioritized_size = batch_size - uniform_size

        uniform_indices = list(
            np.random.randint(self.buffer_counter, size=uniform_size)
            + self.first_leaf_index
        )

        targets = np.random.uniform(size=prioritized_size) * self.sum_tree[0]
        prioritized_indices = [self.search_tree(target) for target in targets]

        indices = np.asarray(uniform_indices + prioritized_indices)
        priorities = np.asarray([self.sum_tree[index] for index in indices])
        assert len(indices) == len(priorities) == batch_size

        uniform_probs = np.asarray(1.0 / self.buffer_counter)
        prioritized_probs = priorities / self.sum_tree[0]

        usp = self.uniform_sample_prob
        sample_probs = (1.0 - usp) * prioritized_probs + usp * uniform_probs
        weights = (uniform_probs / sample_probs) ** beta
        weights /= np.max(weights)
        batch = [self.buffer[idx] for idx in indices - self.first_leaf_index]

        transitions = self.stack_transition(batch)

        sampled_p = np.mean(priorities)
        mean_p = self.sum_tree[0] / self.buffer_counter
        return transitions, weights, indices, sampled_p, mean_p

    @property
    def size(self):
        return self.buffer_counter
