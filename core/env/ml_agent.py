from mlagents.envs import UnityEnvironment
import numpy as np 
import platform, subprocess
from .base import BaseEnv

def match_build():
    os = platform.system()
    if os == "Linux":
        return 'Server' if subprocess.getoutput("which Xorg") == '' else 'Linux'
    else:
        return {"Windows": "Windows",
                "Darwin" : "Max"}[os]

class MLAgent(BaseEnv):
    def __init__(self, env_name, train_mode=True, id=0):
        file_name = f"./core/env/mlagents/{env_name}/{match_build()}/{env_name}"
        self.env = UnityEnvironment(file_name=file_name, worker_id=np.random.randint(60000), seed=id)

        self.train_mode = train_mode
        self.score = 0
        self.default_brain = self.env.brain_names[0]
        self.brain = self.env.brains[self.default_brain]
        self.env_info = self.env.reset(train_mode=train_mode)[self.default_brain]

    def reset(self):
        self.score = 0
        self.env_info = self.env.reset(train_mode=self.train_mode)[self.default_brain]
        state = self.env_info.vector_observations
        return state

    def step(self, action):
        self.env_info = self.env.step(action)[self.default_brain]
        
        next_state = self.env_info.vector_observations
        reward = self.env_info.rewards
        done = self.env_info.local_done
        
        self.score += reward[0] 
        reward, done = map(lambda x: np.expand_dims(x, 0), [reward, done])
        return (next_state, reward, done)

    def close(self):
        self.env.close()

class HopperMLAgent(MLAgent):
    def __init__(self, **kwargs):
        env_name = "Hopper"
        super(HopperMLAgent, self).__init__(env_name, **kwargs)

        self.state_size = 19*4
        self.action_size = 3
        
class PongMLAgent(MLAgent):
    def __init__(self, **kwargs):
        env_name = "Pong"
        super(PongMLAgent, self).__init__(env_name, **kwargs)
        
        self.state_size = 8*4
        self.action_size = 3