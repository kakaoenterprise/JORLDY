import argparse

import torch.multiprocessing as mp

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
                    'action_size': env.action_size}
    agent_config.update(config.agent)
    if config.train.distributed_batch_size:
        agent_config["batch_size"] = config.train.distributed_batch_size
    agent = Agent(**agent_config)
    
    if config.train.load_path:
        agent.load(config.train.load_path)

    trans_queue = mp.Queue()
    interact_sync_queue = mp.Queue(1)
    result_queue = mp.Queue()
    manage_sync_queue = mp.Queue(1)
    
    record_period = config.train.record_period if config.train.record_period else config.train.run_step//10
    test_manager_config = (Env(**config.env), config.train.test_iteration, config.train.record, record_period)
    log_id = config.train.id if config.train.id else config.agent.name
    log_manager_config = (config.env.name, log_id, config.train.experiment)
    agent_config['device'] = "cpu"
    manage = mp.Process(target=manage_process,
                        args=(Agent, agent_config, result_queue, manage_sync_queue,
                              config.train.run_step, config.train.print_period, config.train.save_period,
                              MetricManager, TestManager, test_manager_config,
                              LogManager, log_manager_config, config_manager))
    distributed_manager_config = (Env, config.env, Agent, agent_config, config.train.num_worker)
    interact = mp.Process(target=interact_process,
                            args=(DistributedManager, distributed_manager_config,
                                  trans_queue, interact_sync_queue,
                                  config.train.run_step, config.train.update_period))
    manage.start()
    interact.start()
    try:
        step, print_stamp = 0, 0
        while step < config.train.run_step:
            step += config.train.update_period
            print_stamp += config.train.update_period
            try: interact_sync_queue.get_nowait()
            except: pass
            interact_sync_queue.put(agent.sync_out())
            transitions = trans_queue.get()
            result = agent.process(transitions, step)
            result_queue.put((step, result))
            if print_stamp >= config.train.print_period or step >= config.train.run_step:
                try: manage_sync_queue.get_nowait()
                except: pass
                manage_sync_queue.put(agent.sync_out())
                print_stamp = 0
    except Exception as e:
        traceback.print_exc()
        interact.terminate()
        manage.terminate()
    else:
        print("Optimize process done.")
        interact.join()
        print("Interact process done.")
        manage.join()
        print("Manage process done.")
    finally:
        trans_queue.close()
        interact_sync_queue.close()
        result_queue.close()
        manage_sync_queue.close()
        env.close()