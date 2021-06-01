# RL Algorithms

# Implementation List
### Algorithms

- [Deep Q Network (DQN)](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
- [Double DQN](https://arxiv.org/abs/1509.06461)
- [Dueling DQN](https://arxiv.org/abs/1511.06581)
- [Prioritized Experience Replay (PER)](https://arxiv.org/abs/1511.05952)
- [C51](https://arxiv.org/abs/1707.06887)
- [Noisy](https://arxiv.org/abs/1706.10295)
- [Quantile Regression DQN (QRDQN)](https://arxiv.org/abs/1710.10044)
- [Implicit Quantile Network (IQN)](https://arxiv.org/abs/1806.06923)
- [Rainbow [DQN, IQN]](https://arxiv.org/abs/1710.02298)
- [REINFORCE [Discrete, Continuous]](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)
- [Proximal Policy Optimization (PPO) [Discrete, Continuous]](https://arxiv.org/abs/1707.06347)
- [Soft Actor Critic (SAC) [Continuous]](https://arxiv.org/abs/1801.01290)

### Environments

- Gym (Cartpole, Pendulum, Mountain Car)
- Atari (Alien, Asterix, Assault, Breakout, CrazyClimber, MontezumaRevenge, Pong, PrivateEye, Seaquest, Spaceinvaders) 
- ML-Agents  (Hopper, Pong)



# Install

```
 $ git clone https://github.kakaocorp.com/leonard-q/RL_Algorithms.git  
 $ pip install -r requirements.txt  
 $ python main.py
```



# Results

## Pong (DQN)

<img src="./img/pong_mlagent_score.png" alt="pong_mlagent_score" width=40%/>  <img src="./img/pong_result.gif" alt="pong_result" width=40%/>

## BreakOut (DQN)

<img src="./img/breakout_score.png" alt="breakout_score" width=40%/>  <img src="./img/breakout_result.gif" alt="breakout_result" width=20%/>

## Hopper (SAC)

<img src="./img/hopper_mlagent_score.png" alt="hopper_mlagent_score" width=40%/>  <img src="./img/hopper_result.gif" alt="hopper_result" width=40%/>
