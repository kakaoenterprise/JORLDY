class Manager:
    def __init__(self):
        self.initialized = False
        self.history = dict()
        
    def append(self, result):
        if not self.initialized:
            for key in result.keys():
                self.history[key] = []
            self.initialized = True
        for key in result.keys():
            self.history[key].append(result[key])
    
    def get_statistics(self, mode='mean'):
        ret = dict()
        if mode == 'mean':
            for key in self.history.keys():
                item = self.history[key]
                ret[key] = 0 if len(item) == 0 else round(sum(item)/len(item), 4)
                self.history[key].clear()
        return ret

