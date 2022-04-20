import numpy as np

from .base import BaseBuffer

# Reference: https://github.com/LeejwUniverse/following_deepmid/tree/master/jungwoolee_pytorch/100%20Algorithm_For_RL/01%20sum_tree
class MuzeroPERBuffer(BaseBuffer):
    def __init__(self, buffer_size, uniform_sample_prob=1e-3):
        super(MuzeroPERBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.buffer_counter = 0
        self.tree_size = (self.buffer_size * 2) - 1
        self.first_leaf_index = self.buffer_size - 1

        self.tree_start = self.first_leaf_index
        self.tree_end = self.first_leaf_index
        self.sum_tree = np.zeros(self.tree_size)  # define sum tree
        self.look_up = np.zeros((buffer_size, 2), dtype=int)

        self.trajectories = []
        self.traj_index = 0
        self.traj_offset = 0

        self.max_priority = 1.0
        self.uniform_sample_prob = uniform_sample_prob

    def store(self, transitions):
        if self.first_store and transitions:
            self.check_dim(transitions[0])

        for transition in transitions:
            # TODO: if priority is None

            num = len(transition["priorities"])
            assert num < self.buffer_size

            for i, new_priority in enumerate(transition["priorities"]):
                self.add_tree_data(new_priority, i)

            self.trajectories.append(transition["trajectory"])
            self.traj_index += 1
            self.buffer_counter = min(self.buffer_counter + num, self.buffer_size)

        self.remove_to_fit()

    def add_tree_data(self, new_priority, pos):
        self.update_priority(new_priority, self.tree_end)
        self.look_up[self.tree_end - self.first_leaf_index] = (self.traj_index, pos)

        self.tree_end += 1
        if self.tree_end == self.tree_size:
            self.tree_end = self.first_leaf_index

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

    def remove_to_fit(self):
        if self.buffer_counter < self.buffer_size:
            return

        self.tree_start = self.tree_end
        new_offset, pos = self.look_up[self.tree_end - self.first_leaf_index]
        if pos > 0:
            n_traj = len(self.trajectories[new_offset - self.traj_offset]["rewards"])
            new_start = self.tree_end + n_traj - pos - 1
            if new_start >= self.tree_size:
                self.remove_priorites(self.tree_start, self.tree_size)
                self.tree_start = self.first_leaf_index
                new_start -= self.buffer_size

            self.remove_priorites(self.tree_start, new_start)
            self.tree_start = new_start
            new_offset += 1

        del self.trajectories[: new_offset - self.traj_offset]
        self.traj_offset = new_offset

    def remove_priorites(self, start, end):
        for i in range(start, end):
            self.update_priority(0, i)

        self.buffer_counter -= max(0, end - start)

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
        # TODO: reduce sampling calculation
        uniform_sampling = np.random.uniform(size=batch_size) < self.uniform_sample_prob
        uniform_size = np.sum(uniform_sampling)
        prioritized_size = batch_size - uniform_size

        targets = np.random.randint(
            self.tree_start, self.tree_start + self.buffer_counter, size=uniform_size
        )
        uniform_indices = list(
            np.where(targets < self.tree_size, targets, targets - self.buffer_size)
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

        transitions = [
            (self.trajectories[traj_idx - self.traj_offset], start)
            for traj_idx, start in self.look_up[indices - self.first_leaf_index]
        ]

        sampled_p = np.mean(priorities)
        mean_p = self.sum_tree[0] / self.buffer_counter
        return transitions, weights, indices, sampled_p, mean_p

    def check_dim(self, transition):
        print("########################################")
        print("You should check dimension of transition")
        for key, val in transition["trajectory"].items():
            if len(val) > 1:
                val = val[0]
            print(f"{key}: {val.shape}")
        print("########################################")
        self.first_store = False

    @property
    def size(self):
        return self.buffer_counter
