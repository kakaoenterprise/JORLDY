import numpy as np
from bisect import bisect_left

from .base import BaseBuffer


def epsilon(a, b):
    return abs(a - b) < 1e-6


class MuzeroPERBuffer(BaseBuffer):
    def __init__(self, buffer_size, uniform_sample_prob=1e-3):
        super(MuzeroPERBuffer, self).__init__()
        self.trajectories = []
        self.traj_start = 0
        self.traj_end = 0

        self.buffer_size = buffer_size
        self.buffer_counter = 0
        self.buffer = np.zeros((buffer_size, 2), dtype=int)
        self.priorities = np.zeros((buffer_size, 2))
        self.tree_start = 0
        self.tree_end = 0
        self.sum_cur = 0.0
        self.sum_start = 0.0
        self.sum_last = 0.0
        self.sum_priority = 0.0

        self.uniform_sample_prob = uniform_sample_prob

    def check_dim(self, transition):
        print("########################################")
        print("You should check dimension of transition")
        for key, val in transition["trajectory"].items():
            if len(val) > 1:
                val = val[0]
            print(f"{key}: {val.shape}")
        print("########################################")
        self.first_store = False

    def store(self, transitions):
        if self.first_store:
            self.check_dim(transitions[0])

        for transition in transitions:
            # TODO: if priority is None
            self.add_trajectory(transition["trajectory"], transition["priorities"])

    def add_trajectory(self, trajectory, priorities):
        self.trajectories.append(trajectory)

        num_step = len(priorities)
        new_end = num_step + self.tree_end
        sum_last = self.sum_last
        assert num_step <= self.buffer_size

        if new_end <= self.buffer_size:
            remove_traj, end_step = self.update_priority(
                self.tree_end, new_end, priorities, range(num_step)
            )
        else:
            cut = self.buffer_size - self.tree_end
            self.update_priority(
                self.tree_end, self.buffer_size, priorities[:cut], range(cut)
            )
            self.tree_end, self.sum_cur = 0, 0.0
            self.sum_last = self.priorities[-1].sum()

            new_end -= self.buffer_size
            remove_traj, end_step = self.update_priority(
                0, new_end, priorities[cut:num_step], range(cut, num_step)
            )

        if self.buffer_size < self.buffer_counter:
            remove_end = remove_traj - self.traj_start
            remove_end_step = len(self.trajectories[remove_end]["values"]) - 2
            new_start = (new_end + remove_end_step - end_step) % self.buffer_size

            if new_start < self.tree_start:
                self.sum_priority -= sum_last - self.sum_start
                self.buffer_counter -= self.buffer_size - self.tree_start
                self.tree_start, self.sum_start = 0, 0.0

            bb = self.tree_start
            self.buffer_counter -= new_start - bb
            self.tree_start = new_start

            new_sum_start = self.priorities[self.tree_start][1]
            self.sum_priority -= new_sum_start - self.sum_start
            self.sum_start = new_sum_start

            self.traj_start += remove_end + 1
            del self.trajectories[: remove_end + 1]

        self.tree_end = new_end
        self.traj_end += 1
        assert self.buffer_counter <= self.buffer_size

    def update_priority(self, start, end, priorities, indices):
        new_sum = self.sum_cur
        for i, priority in zip(range(start, end), priorities):
            self.priorities[i, 0] = priority
            self.priorities[i, 1] = new_sum
            new_sum += priority

        self.buffer_counter += len(priorities)
        self.sum_priority += new_sum - self.sum_cur
        self.sum_cur = new_sum
        remove_traj, end_step = self.buffer[end - 1]
        self.buffer[start:end, 0] = self.traj_end
        self.buffer[start:end, 1] = indices

        return remove_traj, end_step

    def search_tree(self, target):
        if self.tree_start < self.tree_end:
            start, end = self.tree_start, self.tree_end
            target += self.sum_start
        elif target > self.sum_cur:
            start, end = self.tree_start, self.buffer_size
            target += self.sum_start - self.sum_cur
        else:
            start, end = 0, self.tree_end

        # TODO: refine BTS
        return start + bisect_left(self.priorities[start:end, 1], target) - 1

    def sample(self, beta, batch_size):
        assert self.sum_priority > 0.0
        # TODO: reduce sampling calculation
        uniform_sampling = np.random.uniform(size=batch_size) < self.uniform_sample_prob
        uniform_size = np.sum(uniform_sampling)
        prioritized_size = batch_size - uniform_size

        indices = np.random.randint(self.buffer_counter, size=batch_size)

        targets = np.random.uniform(size=prioritized_size) * self.sum_priority
        for i, target in zip(range(uniform_size, batch_size), targets):
            indices[i] = self.search_tree(target)

        priorities = np.asarray([self.priorities[index, 0] for index in indices])
        assert len(indices) == len(priorities) == batch_size

        uniform_probs = np.asarray(1.0 / self.buffer_counter)
        prioritized_probs = priorities / self.sum_priority

        usp = self.uniform_sample_prob
        sample_probs = (1.0 - usp) * prioritized_probs + usp * uniform_probs
        weights = (uniform_probs / sample_probs) ** beta
        weights /= np.max(weights)
        trajectories = [
            self.trajectories[self.buffer[index, 0] - self.traj_start]
            for index in indices
        ]
        starts = [self.buffer[index, 1] for index in indices]

        sampled_p = np.mean(priorities)
        mean_p = self.sum_priority / self.buffer_counter
        return trajectories, starts, weights, indices, sampled_p, mean_p

    @property
    def size(self):
        return self.buffer_counter
