# How to use

#### Contents

- [Run Mode Description](#Run-Mode-description)

- [How to Check Implemented List](#How-to-Check-Implemented-List)
  - [Agents](#Agents)
  - [Environments](#Environments)
  - [Networks](#Networks)
- [Run Command Example](#Run-Command-Example)
- [Inference](#Inference)
  - [Saved Files](#Saved-Files)
  - [How to add data in Tensorboard](#How-to-add-data-in-Tensorboard)
  - [How to load trained model](#How-to-load-trained-model)
- [Additional Features](#Additional-Features)
  - [How to use multi-modal structure](#How-to-use-multi-modal-structure)

## Run Mode Description

- single_train: train with single agent.
  - keyword: --single (This keyword can be omitted.)
- sync_distributed_train: train with sychronous distributed setting.
  - keyword: --sync
- async_distributed_train: train with asychronous distributed setting.
  - keyword: --async
- eval: evaluate with trained agent.
  - keyword: --eval

if you want to know the specific process of each script, please refer to [Distributed Architecture](./Distributed_Architecture.md)

## How to Check Implemented List 
- In order to use the various agents(algorithms), environments, and networks provided by JORLDY, you need to know the name that calls the algorithm. JORLDY lists the names of the provided agent, env and network in **_agent_dict.txt**,  **_env_dict.txt** and  **_network_dict.txt**, respectively. 
- **_class_dict.txt** file shows *(key, class)*. You can call the desired element by writing this key to the config file.
- **Note**: If you implement a new environment, agent, or network according to the our documentation and run **main.py**, **_class_dict.txt** will be updated automatically.

### Agents
- A list of implemented agents can be found in [_agent_dict.txt](../jorldy/core/agent/_agent_dict.txt).

- Example: You can check the key of the Ape-X agent in [_agent_dict.txt](../jorldy/core/agent/_agent_dict.txt): *('ape_x', <class 'core.agent.ape_x.ApeX'>)*. If you want to use the Ape-X agent, write agent.name as *ape_x* in config file.

```python
agent = {
    "name": "ape_x",
    "network": "dueling",
    ...
}
```

### Environments
- Provided environments list can be found in [_env_dict.txt](../jorldy/core/env/_env_dict.txt).

- Example: You can check the key of the Procgen starpilot environment in [_env_dict.txt](../jorldy/core/env/_env_dict.txt): *('starpilot', <class 'core.env.procgen.Starpilot'>)*. If you want to use the starpilot environment, it should be defined in the command using the key of starpilot environment. ex) python main.py --config config.dqn.procgen --env.name starpilot.

### Networks
- A list of implemented networks can be found in [_network_dict.txt](../jorldy/core/network/_network_dict.txt).
- If the network you want to use requires a head element, you should also include head in the config. A list of implemented heads can be found in [_head_dict.txt](../jorldy/core/network/_head_dict.txt).
- **Note**: To use head in your customized network, you should inherit the [BaseNetwork class](../jorldy/core/network/base.py). We refer to [How to customize network](../jorldy/core/network/README.md).

- Example 1: You can check the key of the PPO discrete policy network in [_network_dict.txt](../jorldy/core/network/_network_dict.txt): *('discrete_policy_value', <class 'core.network.policy_value.DiscretePolicyValue'>)*. If you want to use the PPO discrete policy network, write agent.network as *discrete_policy_value* in config file. 
```python
agent = {
    "name":"ppo",
    "network":"discrete_policy_value",
    ...
}
```

- Example 2: Use head case (image state); add the key "head" and set the value "cnn" to the agent dictionary in config file.

```python
agent = {
    "name":"ppo",
    "network":"discrete_policy_value",
    "head": "cnn",
    ...
}
```
## Run Command Example 
- Default command line consists of run_mode part and config part. When you type __*config path*__, you should omit '.py' in the name of the config file. If you do not type __*config path*__, It runs with the default config in the main.py. Similarly, if you do not type __*run mode*__, it run as single_train.
    ```
    python main.py --config [config path]
    ```
    - Example:
        ``` 
        # single_train
        python main.py --config config.dqn.cartpole 
        ```
- If you want to use a __*run mode*__, type a keyword
    ```
    python main.py --[run mode] --config [config path]
    ```
    - Example:
        ``` 
        python main.py --sync --config config.ppo.cartpole 
        ```
- If you want to load environment in the atari (or procgen), use the atari (or procgen) config path and define environment by using the parser env.name. 
    ```
    python main.py --[run mode] --config [config path] --env.name [env name]
    ```
    - Example:
        ``` 
        python main.py --single --config config.dqn.atari --env.name assault 
        ```
- All parameters in the config file can be changed by using the parser without modifying the config file.
    ```
    python main.py --[run mode] --config [config path] --[optional parameter key] [optional parameter value]
    ```
    - Example:
        ``` 
        python main.py --single --config config.dqn.cartpole --agent.batch_size 64 
        ```
        ``` 
        python main.py --sync --config config.ppo.cartpole --train.num_worker 8 
        ```

- Executable run mode list: **single_train**: --single, **sync_distributed_train**: --sync, **async_distributed_train**: --async,
**eval**: --eval.



## Inference 

### Saved Files 

- The files are saved in the path **logs/[env name]/[Algorithm]/[Datetime]/**
  - Ex) logs/breakout/rainbow/20211014152800
- The saved files are as follows 
  - **gif files**: gif of test episode
  - **ckpt**: saved Pytorch checkpoint file 
  - **config.py**: configuration of the running 
  - **events.out.tfevents...**: saved TensorBoard event file

### How to add data in Tensorboard

- The TensorBoard data can be added by modifying the **core/agent** codes
- For example, Noisy algorithm adds mean value of the sigma to the Tensorboard. To do this, add sigma to the result dictionary inside the process function of the agent as follows.

```python
result = {
"loss" : loss.item(),
"max_Q": max_Q,
"sig_w1": sig1_mean.item(),
"sig_w2": sig2_mean.item(),
}
```

- If you check the TensorBoard after performing the above process, you can see that the sigma values are added as follows.

<img src="../resrc/noisy_tensorboard.png" alt="noisy_tensorboard" width=100%/> 

### How to load trained model

- If you want to load the trained model, you should set path of the saved model in the train part in config. 
  - If the saved model is not loaded, set "load path" as "None"
- Example 
  - env: space invaders
  - algorithm: rainbow (if you want to test the model without training, set "training" as False) 

```python
train = {
    "training" : False,
    "load_path" : "./logs/spaceinvaders/rainbow/20211015110908/",
		...
}
```



## Additional Features

### How to use multi-modal structure

- JORLDY supports multi-modal structured networks which uses both images and vector input data. 

**To run the agents with multi-modal input**

- The state size has to be set as list consist of image and vector shape :arrow_right: state_size = [image shape, vector shape]

  - e.g. if image has shape [4,84,84] and length of vector is 10 the state size should be [[4,84,84], 10]

- Also, the state has to be set as list consist of image and vector data :arrow_right: state = [image data, vector data]

- In the config files, "head" in agent part should be set as "multi" as follows

  ```python
  agent = {
      "name": "ppo",
      "network": "continuous_policy_value",
      "head": "multi",
      "gamma": 0.99,
      ...
  }
  ```

- It's done! You can train the agent, which uses both images and vector data as an input. :sunglasses:
