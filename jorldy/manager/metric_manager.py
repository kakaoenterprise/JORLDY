from collections import defaultdict


class MetricManager:
    def __init__(self):
        self.metrics = defaultdict(int)
        self.counters = defaultdict(int)

    def append(self, result):
        for key, value in result.items():
            c = self.counters[key]
            self.metrics[key] = (self.metrics[key] * (c / (c + 1))) + (value / (c + 1))
            self.counters[key] += 1

    def get_statistics(self):
        ret = dict()
        for key, value in self.metrics.items():
            ret[key] = round(value, 4)
        self.metrics.clear()
        self.counters.clear()
        return ret
