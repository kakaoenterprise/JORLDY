from core import *
from managers import *
from process import *

# import config.YOUR_AGENT.YOUR_ENV as config
import config.icm_ppo.alien as config
import torch.multiprocessing as mp

if __name__=="__main__":
    env = Env(**config.env)
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

    result_queue = mp.Queue()
    manage_sync_queue = mp.Queue(1)
    
    test_manager_config = (config.train["test_iteration"],)
    log_id = config.agent["name"] if "id" not in config.train.keys() else config.train["id"]
    purpose = None if "purpose" not in config.train.keys() else config.train["purpose"]
    log_manager_config = (config.env["name"], log_id, purpose)
    agent_config['device'] = "cpu"
    manage = mp.Process(target=manage_process,
                        args=(Agent, agent_config, Env(**config.env), result_queue, manage_sync_queue,
                              run_step, print_period, save_period, MetricManager,
                              TestManager, test_manager_config,
                              LogManager, log_manager_config))
    manage.start()
    state = env.reset()
    try:
        for step in range(1, run_step+1):
            action = agent.act(state, training=True)
            next_state, reward, done = env.step(action)

            result = agent.process([(state, action, reward, next_state, done)], step)
            result_queue.put((step, result))

            if step % print_period == 0 or step == run_step:
                try: manage_sync_queue.get_nowait()
                except: pass
                manage_sync_queue.put(agent.sync_out())
            
            state = next_state if not done else env.reset() 
    except Exception as e:
        traceback.print_exc()
        manage.terminate()
    else:
        print("Optimize process done.")
        manage.join()
        print("Manage process done.")
    finally:
        result_queue.close()
        manage_sync_queue.close()
        env.close()
