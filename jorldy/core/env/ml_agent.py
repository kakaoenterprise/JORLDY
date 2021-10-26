from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import numpy as np 
import platform, subprocess
from .base import BaseEnv

def match_build():
    os = platform.system()
    if os == "Linux":
        return 'Server' if subprocess.getoutput("which Xorg") == '' else 'Linux'
    else:
        return {"Windows": "Windows",
                "Darwin" : "Mac"}[os]

class _MLAgent(BaseEnv):
    def __init__(self, env_name, train_mode=True, id=None):
        env_path = f"./core/env/mlagents/{env_name}/{match_build()}/{env_name}"
        id = np.random.randint(65534) if id is None else id
                
        engine_configuration_channel = EngineConfigurationChannel()
        self.env = UnityEnvironment(file_name=env_path,
                                    side_channels=[engine_configuration_channel],
                                    worker_id=id)
        
        self.env.reset()
        
        self.train_mode = train_mode
        self.score = 0
                
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.spec = self.env.behavior_specs[self.behavior_name]
        
        self.is_continuous_action = self.spec.action_spec.is_continuous()
        
        engine_configuration_channel.set_configuration_parameters(time_scale=12.0)
        dec, term = self.env.get_steps(self.behavior_name)
        
    def reset(self):
        self.score = 0
        self.env.reset()
        dec, term = self.env.get_steps(self.behavior_name)
        state = dec.obs[0]

        return state

    def step(self, action):
        action_tuple = ActionTuple()
        
        if self.is_continuous_action:
            action_tuple.add_continuous(action)
        else:
            action_tuple.add_discrete(action)
        
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()
        
        dec, term = self.env.get_steps(self.behavior_name)
        done = len(term.agent_id) > 0
        reward = term.reward if done else dec.reward
        next_state = term.obs[0] if done else dec.obs[0]
        
        self.score += reward[0] 

        reward, done = map(lambda x: np.expand_dims(x,0), [reward, [done]])
        
        return (next_state, reward, done)

    def close(self):
        self.env.close()

class HopperMLAgent(_MLAgent):
    def __init__(self, **kwargs):
        env_name = "Hopper"
        super(HopperMLAgent, self).__init__(env_name, **kwargs)

        self.state_size = 19*4
        self.action_size = 3
        
class PongMLAgent(_MLAgent):
    def __init__(self, **kwargs):
        env_name = "Pong"
        super(PongMLAgent, self).__init__(env_name, **kwargs)
        
        self.state_size = 8*1
        self.action_size = 3
        
class WormMLAgent(_MLAgent):
    def __init__(self, **kwargs):
        env_name = "Worm"
        super(WormMLAgent, self).__init__(env_name, **kwargs)

        self.state_size = 64*1
        self.action_size = 9