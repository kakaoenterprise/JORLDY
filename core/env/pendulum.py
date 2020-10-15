import gym

class Pendulum:
    def __init__(self, mode='continuous'):
        self.env = gym.make('Pendulum-v0')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.score = 0

    def reset(self):
        self.score = 0
        state = self.env.reset()
        return state

    def step(self, action):
        next_state, reward, done, info = self.env.step([action])
        self.score += reward 
        return (next_state, reward, done)

    def close(self):
        self.env.close()
    
