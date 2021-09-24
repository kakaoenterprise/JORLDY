import numpy as np 

class TestManager:
    def __init__(self, env, iteration=10, record=None, record_period=None):
        self.env = env
        self.iteration = iteration if iteration else 10
        assert iteration > 0
        self.record = record and env.recordable()
        self.record_period = record_period
        self.record_stamp = 0
        self.time_t = 0
    
    def test(self, agent, step):
        scores = []
        frames = []
        self.record_stamp += step - self.time_t
        self.time_t = step
        record = self.record and self.record_stamp >= self.record_period
        
        for i in range(self.iteration):
            done = False
            state = self.env.reset()
            while not done:
                # record first iteration
                if record and i == 0: 
                    frames.append(self.env.get_frame())
                action_dict = agent.act(state, training=False)
                next_state, reward, done = self.env.step(action_dict['action'])
                transition = {'state': state, 'next_state': next_state,
                              'reward': reward, 'done': done}
                transition.update(action_dict)
                agent.interact_callback(transition)
                state = next_state
            scores.append(self.env.score)
            
        if record:
            self.record_stamp = 0
        return np.mean(scores), frames