# How to customize environment

## 1. Inherit BaseEnv class.
- If you want to implement a new environment without inheriting the provided environments, you must inherit the base environment.
- Every environment must includes keward arguments when defined. You should use \*\*kwargs in \_\_init\_\_.

reference: [base.py](./base.py), [gym_env.py](./gym_env.py), [atari.py](./atari.py), ...

## 2. Implement abstract methods.
- Abstract methods(__reset__, __step__, __close__) should be implemented. Implement these methods by referring to the comments.
- When you implement a __step__ method, you should expand dimension of state, reward, and done from (d) to (1,d) using __expand_dim__.

reference: [gym_env.py](./gym_env.py)

## 3. If necessary, implement another method.
- __recordable__ indicates the environment can be recorded as a gif. If a newly implemented environment has no visual state, set __recordable__ to return False. 
- If you want to record evaluation episode as a gif file, make sure __recordable__ returns True and set train.record in config file to True.

reference: [atari.py](./atari.py), [procgen.py](./procgen.py), [nes.py](./nes.py)

## When adding open source environment.
- If you want to add an open source environment, we refer to some provided environments; atari, procgen, and nes environments.

reference: [atari.py](./atari.py), [procgen.py](./procgen.py), [nes.py](./nes.py)
