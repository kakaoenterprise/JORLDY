from core import *
from manager import *
from process import *

# import config.YOUR_AGENT.YOUR_ENV as config
import config.ppo.breakout as config
import torch.multiprocessing as mp

if __name__ == '__main__':
    env = Env(**config.env)
    config.agent["batch_size"] *= config.train["num_worker"]
    agent_config = {'state_size': env.state_size,
                    'action_size': env.action_size}
    agent_config.update(config.agent)
    agent = Agent(**agent_config)
    
    load_path = config.train["load_path"]
    if load_path:
        agent.load(load_path)

    run_step = config.train["run_step"]
    print_period = config.train["print_period"]
    save_period = config.train["save_period"]
    update_period = config.train["update_period"]

    trans_queue = mp.Queue()
    interact_sync_queue = mp.Queue(1)
    result_queue = mp.Queue()
    manage_sync_queue = mp.Queue(1)
    
    record = False if "record" not in config.train.keys() else config.train["record"]
    record_period = run_step//10 if "record_period" not in config.train.keys() else config.train["record_period"]
    test_manager_config = (Env(**config.env), config.train["test_iteration"], record, record_period)
    log_id = config.agent["name"] if "id" not in config.train.keys() else config.train["id"]
    purpose = None if "purpose" not in config.train.keys() else config.train["purpose"]
    log_manager_config = (config.env["name"], log_id, purpose)
    agent_config['device'] = "cpu"
    manage = mp.Process(target=manage_process,
                        args=(Agent, agent_config, result_queue, manage_sync_queue,
                              run_step, print_period, save_period, MetricManager,
                              TestManager, test_manager_config,
                              LogManager, log_manager_config))
    distributed_manager_config = (Env, config.env, Agent, agent_config, config.train["num_worker"])
    interact = mp.Process(target=interact_process,
                            args=(DistributedManager, distributed_manager_config,
                                  trans_queue, interact_sync_queue, run_step, update_period))
    manage.start()
    interact.start()
    try:
        step, print_stamp = 0, 0
        while step < run_step:
            step += update_period
            print_stamp += update_period
            try: interact_sync_queue.get_nowait()
            except: pass
            interact_sync_queue.put(agent.sync_out())
            transitions = trans_queue.get()
            result = agent.process(transitions, step)
            result_queue.put((step, result))
            if print_stamp >= print_period or step >= run_step:
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