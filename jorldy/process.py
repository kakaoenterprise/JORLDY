import traceback
import time

# Interact (Async)
def interact_process(DistributedManager, distributed_manager_config,
                     trans_queue, sync_queue, run_step, update_period):
    distributed_manager = DistributedManager(*distributed_manager_config)
    num_workers = distributed_manager.num_workers
    step = 0
    try:
        while step < run_step:
            transitions = distributed_manager.run(update_period)
            delta_t = len(transitions) / num_workers
            step += delta_t
            trans_queue.put((int(step), transitions))
            if sync_queue.full():
                distributed_manager.sync(sync_queue.get())
            while trans_queue.full():
                time.sleep(0.1)
    except Exception as e:
        traceback.print_exc()
    finally:
        distributed_manager.terminate()
        
# Manage
def manage_process(Agent, agent_config,
                   result_queue, sync_queue, path_queue,
                   run_step, print_period, MetricManager,
                   EvalManager, eval_manager_config,
                   LogManager, log_manager_config, config_manager):
    agent = Agent(**agent_config)
    eval_manager = EvalManager(*eval_manager_config)
    metric_manager = MetricManager()
    log_manager = LogManager(*log_manager_config)
    path_queue.put(log_manager.path)
    config_manager.dump(log_manager.path)
    
    step, print_stamp = 0, 0
    try:
        while step < run_step:
            wait = True
            while wait or not result_queue.empty():
                _step, result = result_queue.get()
                metric_manager.append(result)
                wait = False
            print_stamp += _step - step
            step = _step
            if print_stamp >= print_period or step >= run_step: 
                agent.sync_in(**sync_queue.get())
                score, frames = eval_manager.evaluate(agent, step)
                metric_manager.append({"score": score})
                statistics = metric_manager.get_statistics()
                print(f"Step : {step} / {statistics}")
                log_manager.write(statistics, frames, score, step)
                print_stamp = 0
    except Exception as e:
        traceback.print_exc()