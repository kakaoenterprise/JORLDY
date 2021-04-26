from core import *
from managers import *

# import config.YOUR_AGENT.YOUR_ENV as config
import config.ppo.cartpole as config
import torch.multiprocessing as mp

# Interact
def interact_process(DistributedManager, distributed_manager_config,
                     trans_queue, sync_queue, run_step, update_term):
    distributed_manager = DistributedManager(*distributed_manager_config)
    step = 0
    try:
        while step < run_step:
            step += update_term
            trans_queue.put(distributed_manager.run(update_term))
            distributed_manager.sync(sync_queue.get())
    except Exception as e:
        print(e)
        distributed_manager.terminate()
        
# Manage
def manage_process(agent, env, result_queue, sync_queue,
                   run_step, print_term, save_term, MetricManager,
                   TestManager, test_manager_config,
                   LogManager, log_manager_config):
    test_manager = TestManager(*test_manager_config)
    metric_manager = MetricManager()
    log_manager = LogManager(*log_manager_config)
    
    step = 0
    try:
        while step < run_step:
            step, result = result_queue.get()
            metric_manager.append(result)
            while not result_queue.empty():
                step, result = result_queue.get()
                metric_manager.append(result)
            if step % print_term == 0: #and not sync_queue.empty():
                agent.sync_in(**sync_queue.get())
                score = test_manager.test(agent, env)
                metric_manager.append({"score": score})
                statistics = metric_manager.get_statistics()
                print(f"Step : {step} / {statistics}")
                log_manager.write_scalar(statistics, step)
            if step % save_term == 0 or step >= run_step:
                agent.save(log_manager.path)
    except Exception as e:
        print(e)
            
if __name__ == '__main__':
    env = Env(**config.env)
    config.agent["batch_size"] *= config.train["num_worker"]
    agent = Agent(state_size=env.state_size,
                  action_size=env.action_size,
                  **config.agent)
    
    load_path = config.train["load_path"]
    if load_path:
        agent.load(load_path)

    run_step = config.train["run_step"]
    print_term = config.train["print_term"]
    save_term = config.train["save_term"]
    update_term = config.train["update_term"]

    trans_queue = mp.Queue()
    actor_sync_queue = mp.Queue(1)
    result_queue = mp.Queue()
    test_sync_queue = mp.Queue(1)
    
    test_manager_config = (config.train["test_iteration"],)
    log_id = config.agent["name"] if "id" not in config.train.keys() else config.train["id"]
    log_manager_config = (config.env["name"], log_id)
    manage = mp.Process(target=manage_process,
                        args=(agent.cpu(), env, result_queue, test_sync_queue,
                              run_step, print_term, save_term, MetricManager,
                              TestManager, test_manager_config,
                              LogManager, log_manager_config))
    distributed_manager_config = (Env, config.env, agent.cpu(), config.train["num_worker"])
    interact = mp.Process(target=interact_process,
                            args=(DistributedManager, distributed_manager_config,
                                  trans_queue, actor_sync_queue, run_step, update_term))
    manage.start()
    interact.start()
    try:
        step = 0
        while step < run_step:
            step += update_term
            if actor_sync_queue.full():
                actor_sync_queue.get()
            actor_sync_queue.put(agent.sync_out())
            transitions = trans_queue.get()
            result = agent.process(transitions)
            if result:
                result_queue.put((step, result))
            if step % print_term == 0:
                if test_sync_queue.full():
                    test_sync_queue.get()
                test_sync_queue.put(agent.sync_out())
        interact.join()
        manage.join()
    except Exception as e:
        print(e)
        trans_queue.close()
        actor_sync_queue.close()
        result_queue.close()
        test_sync_queue.close()
        env.close()