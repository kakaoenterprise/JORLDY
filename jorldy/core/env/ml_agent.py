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
    """MLAgent environment. 
    
    Args: 
        env_name (str): name of environment in ML-Agents.
        train_mode (bool): parameter that determine whether to use low-resource training rendering mode.
    """
    def __init__(self, env_name, train_mode=True, id=None, **kwargs):
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
        state = self.state_config(dec.obs)

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
        next_state = self.state_config(term.obs) if done else self.state_config(dec.obs)
        
        self.score += reward[0] 

        reward, done = map(lambda x: np.expand_dims(x,0), [reward, [done]])

        return (next_state, reward, done)

    def state_config(self, obs):
        vec_state = None
        img_state = None 

        for obs_i in obs:
            if len(obs_i.shape) == 2:
                if vec_state is None: 
                    vec_state = obs_i 
                else:
                    vec_state = np.concatenate((vec_state, obs_i), axis=-1)
            else:
                if img_state is None:
                    img_state = obs_i 
                else:
                    img_state = np.concatenate((img_state, obs_i), axis=-1)

        if img_state is not None:
            img_state = np.transpose(img_state, (0,3,1,2))
            if vec_state is None:
                return img_state 
            else:
                return [img_state, vec_state]
        else:
            return vec_state

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

class DroneMLAgent(_MLAgent):
    def __init__(self, **kwargs):
        env_name = "drone"
        super(DroneMLAgent, self).__init__(env_name, **kwargs)

        self.state_size = [[15,36,64], 95]
        self.action_size = 3

class WormMLAgent(_MLAgent):
    def __init__(self, **kwargs):
        env_name = "Worm"
        super(WormMLAgent, self).__init__(env_name, **kwargs)

        self.state_size = 64*1
        self.action_size = 9