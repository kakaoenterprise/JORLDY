# Role of managers

The manager is responsible for the non-learning aspects. The roles of each manager are as follows.

### config_manager
- It processes the config file and the optional parameter of run_command, and dumps the config to the storage path.

### distributed_manager
- It manages actors in distributed scripts. Let actors interact for update_period and sync actors network.

### eval_manager
- A simulation is performed for a certain number of episodes to obtain an evaluated score.
- It saves the frames for recording.

### log_manager
- It records the learning progress in TensorBoard. 
- It receives the frames for recording and creates it as a gif.

### metric_manager
- It manages metrics and calculates and provides statistics.
