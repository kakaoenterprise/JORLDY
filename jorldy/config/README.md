# How to customize config

## Config file management rules
- The config file provided by default is mainly managed in the form of config/\[agent\]/\[env\].py. 
- For a specific environment group that shares parameters, manage it in the form of config/\[agent\]/\[env_group\], and specify the environment name with --env.name in the run command.

reference: [dqn/cartpole.py](./dqn/cartpole.py), [dqn/atari.py](./dqn/atari.py)

## Config setting
- The config file is managed with a total of four dictionary variables: agent, env, optim, and train. 

  ### agent
    - The agent dictionary manages input parameters used by the agent class. 
      - name: The key of the agent class you want to use.
      - others: You can check it in the agent class.
  
  ### env
    - The env dictionary manages input parameters used by the env class. 
      - name: The key of the env class you want to use.
      - others: You can check it in the env class.
  
  ### optim
    - The optim dictionary manages input parameters used by the optimizer class. Since the optimizer of pytorch is used as it is, any optimizer supported by pytorch can be used.
      - name: The key of the optimizer class you want to use.
      - others: You can check it in the optimizer class supported by pytorch.
  
  ### train
    - The optim dictionary manages parameters used in the main script.
      - training: It means whether to learn. Set to False in the eval.py script and True otherwise.
      - load_path: It means the path to load the model. If you want to load the model or in the eval.py script, you need to set it. If not, set it None.
      - run_step: It determines the total number of interactions to proceed.
      - print_period: It means the cycle(unit=step) to print the progress.
      - save_period: It means the cycle(unit=step) to save the model.
      - eval_iteration: It means how many episodes will be run in total to get the evaluation score.
      - record: It means whether to record the simulation as the evaluation proceeds. If you set it True, simulation is saved as a gif file in save_path. If you set it True and env is recordable, simulation is saved as a gif file in save_path. (Note that this does not work for non-recordable environments.)
      - record_period: It means the cycle(unit=step) to record.
      - id: If set, log to logs/\[env\]/__id__ path. (default: __agent.name__)
      - experiment: If set, log to logs/__experiment__/\[env\]/id. Otherwise, log to logs/\[env\]/id.
      - distributed_batch_size: In distributed script, uses distributed_batch_size instead of agent.batch_size.
      - update_period: It means the cycle(unit=step) in which actors pass transition data to learner.
      - num_workers: Total number of distributed actors which interact with env.
      - eval_time_limit: Time limit(unit=seconds) given per episode when evaluating the model. (default: No limit).


      __distributed_batch_size, update_period and num_workers are only used in distributed  scripts.__

reference: [ppo/atari.py](./ppo/atari.py)
