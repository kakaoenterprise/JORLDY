# How to use

## How to Check Implemented List 
- In order to use the various agents(algorithms), environments, and networks provided by JORLDY, you need to know the name that calls the algorithm. JORLDY lists the names of the provided agent, env and network in **_agent_dict.txt**,  **_env_dict.txt** and  **_network_dict.txt**, respectively. 
- **_class_dict.txt** file shows *(key, class)*. You can call the desired element by writing this key to the config file.
- **Note**: If you implement a new environment, agent, or network according to the our documentation and run **main.py**, **_class_dict.txt** will be updated automatically.

### Agents
- A list of implemented agents can be found in [_agent_dict.txt](../core/agent/_agent_dict.txt).

- Example: You can check the key of the Ape-X agent in [_agent_dict.txt](../core/agent/_agent_dict.txt): *('ape_x', <class 'core.agent.ape_x.ApeX'>)*. If you want to use the Ape-X agent, write agent.name as *ape_x* in config file.

```python
agent = {
    "name": "ape_x",
    "network": "dueling",
    ...
}
```

### Environments
- Provided environments list can be found in [_env_dict.txt](../core/env/_env_dict.txt).

- Example: You can check the key of the Procgen starpilot environment in [_env_dict.txt](../core/env/_env_dict.txt): *('starpilot', <class 'core.env.procgen.Starpilot'>)*. If you want to use the starpilot environment, it should be defined in the command using the key of starpilot environment. ex) python main.py --config config.dqn.procgen --env.name starpilot.

### Networks
- A list of implemented networks can be found in [_network_dict.txt](../core/network/_network_dict.txt).
- If the network you want to use requires a head element, you should also include head in the config. A list of implemented heads can be found in [_head_dict.txt](../core/network/_head_dict.txt).
- **Note**: To use head in your customized network, you should inherit the [BaseNetwork class](../core/network/base.py). We refer to [How to customize network](../core/network/README.md).

- Example 1: You can check the key of the PPO discrete policy network in [_network_dict.txt](../core/network/_network_dict.txt): *('discrete_policy_value', <class 'core.network.policy_value.DiscretePolicyValue'>)*. If you want to use the PPO discrete policy network, write agent.network as *discrete_policy_value* in config file. 
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
- Default command line consists of script part and config part. When you type __*config path*__, you should omit '.py' in the name of the config file. If you do not type __*config path*__, It runs with the default config in the script.
    ```
    python [script name].py --config [config path]
    ```
    - Example:
        ``` 
        python single_train.py --config config.dqn.cartpole 
        ```

- If you want to load environment in the atari (or procgen), use the atari (or procgen) config path and define environment by using the parser env.name. 
    ```
    python [script name].py --config [config path] --env.name [env name]
    ```
    - Example:
        ``` 
        python single_train.py --config config.dqn.atari --env.name assault 
        ```
- All parameters in the config file can be changed by using the parser without modifying the config file.
    ```
    python [script name].py --config [config path] --[optional parameter key] [optional parameter value]
    ```
    - Example:
        ``` 
        python single_train.py --config config.dqn.cartpole --agent.batch_size 64 
        ```
        ``` 
        python sync_distributed_train.py --config config.ppo.cartpole --train.num_worker 8 
        ```

- Executable script list: **single_train.py**, **sync_distributed_train.py**, **async_distributed_train.py**.

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

