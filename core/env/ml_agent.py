from mlagents.envs import UnityEnvironment
import numpy as np 

class HopperMLAgent:
    def __init__(self, train_mode=True):
        self.env = UnityEnvironment(file_name="./core/env/mlagent_env/Hopper/MAC/Hopper", worker_id=np.random.randint(65535))
        self.state_size = 19*4
        self.action_size = 3
        self.train_mode = train_mode
        
        self.score = 0
        
        self.default_brain = self.env.brain_names[0]
        self.brain = self.env.brains[self.default_brain]
        
        self.env_info = self.env.reset(train_mode=train_mode)[self.default_brain]
        
    def reset(self):
        self.score = 0
        self.env_info = self.env.reset(train_mode=self.train_mode)[self.default_brain]
        state = np.array(self.env_info.vector_observations[0])
        return state

    def step(self, action):
        self.env_info = self.env.step(action)[self.default_brain]
        
        next_state = np.array(self.env_info.vector_observations[0])
        reward = self.env_info.rewards[0]
        done = self.env_info.local_done[0]
        
        self.score += reward 

        return (next_state, reward, done)

    def close(self):
        self.env.close()

class PongMLAgent:
    def __init__(self, train_mode=True):
        self.env = UnityEnvironment(file_name="./core/env/mlagent_env/Pong/MAC/Pong", worker_id=np.random.randint(65535))
        self.state_size = 8*4
        self.action_size = 3
        self.train_mode = train_mode
        
        self.score = 0
        
        self.default_brain = self.env.brain_names[0]
        self.brain = self.env.brains[self.default_brain]
        
        self.env_info = self.env.reset(train_mode=train_mode)[self.default_brain]
        
    def reset(self):
        self.score = 0
        self.env_info = self.env.reset(train_mode=self.train_mode)[self.default_brain]
        state = np.array(self.env_info.vector_observations[0])
        return state

    def step(self, action):
        self.env_info = self.env.step(action)[self.default_brain]
        
        next_state = np.array(self.env_info.vector_observations[0])
        reward = self.env_info.rewards[0]
        done = self.env_info.local_done[0]
        
        self.score += reward 

        return (next_state, reward, done)

    def close(self):
        self.env.close()