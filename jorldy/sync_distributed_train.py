import argparse

import multiprocessing as mp

from core import *
from manager import *
from process import *

# default_config_path = "config.YOUR_AGENT.YOUR_ENV"
default_config_path = "config.dqn.cartpole"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config.dqn.cartpole')
    args, unknown = parser.parse_known_args()
    config_path = args.config if args.config else default_config_path
    config_manager = ConfigManager(config_path, unknown)
    config = config_manager.config
    
    env = Env(**config.env)
    agent_config = {'state_size': env.state_size,
                    'action_size': env.action_size,
                    'optim_config': config.optim,
                    'num_workers': config.train.num_workers}
    agent_config.update(config.agent)
    if config.train.distributed_batch_size:
        agent_config["batch_size"] = config.train.distributed_batch_size
    agent = Agent(**agent_config)
    
    if config.train.load_path:
        agent.load(config.train.load_path)

    result_queue = mp.Queue()
    manage_sync_queue = mp.Queue(1)
    path_queue = mp.Queue(1)
    
    record_period = config.train.record_period if config.train.record_period else config.train.run_step//10
    eval_manager_config = (Env(**config.env), config.train.eval_iteration, config.train.record, record_period)
    log_id = config.train.id if config.train.id else config.agent.name
    log_manager_config = (config.env.name, log_id, config.train.experiment)
    agent_config['device'] = "cpu"
    manage = mp.Process(target=manage_process,
                        args=(Agent, agent_config,
                              result_queue, manage_sync_queue, path_queue,
                              config.train.run_step, config.train.print_period,
                              MetricManager, EvalManager, eval_manager_config,
                              LogManager, log_manager_config, config_manager))
    
    distributed_manager = DistributedManager(Env, config.env, Agent, agent_config, config.train.num_workers, 'sync')
    
    manage.start()
    try:
        save_path = path_queue.get()
        step, print_stamp, save_stamp = 0, 0, 0
        while step < config.train.run_step:
            transitions = distributed_manager.run(config.train.update_period)
            step += config.train.update_period
            print_stamp += config.train.update_period
            save_stamp += config.train.update_period
            result = agent.process(transitions, step)
            distributed_manager.sync(agent.sync_out())
            result_queue.put((step, result))
            if print_stamp >= config.train.print_period or step >= config.train.run_step:
                try: manage_sync_queue.get_nowait()
                except: pass
                manage_sync_queue.put(agent.sync_out())
                print_stamp = 0
            if save_stamp >= config.train.save_period or step >= config.train.run_step:
                agent.save(save_path)
                save_stamp = 0
    except Exception as e:
        traceback.print_exc()
        manage.terminate()
    else:
        print("Main process done.")
        manage.join()
        print("Manage process done.")
    finally:
        result_queue.close()
        manage_sync_queue.close()
        path_queue.close()
        env.close()