from .dqn import DQNAgent
from .sac import SACAgent

class Agent:
    dictionary = {
    "dqn": DQNAgent,
    "sac": SACAgent,
    }
    
    def __new__(self, name, *args, **kwargs):
        expected_type = str
        if type(name) != expected_type:
            print("### name variable must be string! ###")
            raise Exception
        name = name.lower()
        if not name in self.dictionary.keys():
            print(f"### can use only follows {[opt for opt in self.dictionary.keys()]}")
            raise Exception
        return self.dictionary[name](*args, **kwargs)

'''
class TemplateAgent:
    def __init__(self):
        pass

    def act(self, state):
        pass

    def learn(self):
        pass
    
    def observe(self, state, action, reward, next_state, done)
        # Process per step
        
        # Process per epi
        if done :
            pass
'''