import traceback
import time
from threading import Thread

# Interact (for async distributed train)
def interact_process(
    DistributedManager,
    distributed_manager_config,
    trans_queue,
    sync_queue,
    run_step,
    update_period,
):
    distributed_manager = DistributedManager(*distributed_manager_config)
    num_workers = distributed_manager.num_workers
    step = 0
    try:
        while step < run_step:
            transitions = distributed_manager.run(update_period)
            delta_t = len(transitions) // num_workers
            step = step + delta_t
            trans_queue.put((step, transitions))
            if sync_queue.full():
                distributed_manager.sync(sync_queue.get())
            while trans_queue.full():
                time.sleep(0.1)
    except Exception as e:
        traceback.print_exc()
    finally:
        distributed_manager.terminate()


# Manage
def manage_process(
    Agent,
    agent_config,
    result_queue,
    sync_queue,
    path_queue,
    run_step,
    print_period,
    MetricManager,
    EvalManager,
    eval_manager_config,
    LogManager,
    log_manager_config,
    config_manager,
):
    agent = Agent(**agent_config)
    eval_manager = EvalManager(*eval_manager_config)
    metric_manager = MetricManager()
    log_manager = LogManager(*log_manager_config)
    path_queue.put(log_manager.path)
    config_manager.dump(log_manager.path)

    step, print_stamp, eval_thread = 0, 0, None
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
                if (
                    eval_thread is None
                    or not eval_thread.is_alive()
                    or step >= run_step
                ):
                    if eval_thread is not None:
                        eval_thread.join()
                    agent.sync_in(**sync_queue.get())
                    statistics = metric_manager.get_statistics()
                    eval_thread = Thread(
                        target=evaluate_thread,
                        args=(agent, step, statistics, eval_manager, log_manager),
                    )
                    eval_thread.start()
                print_stamp = 0
    except Exception as e:
        traceback.print_exc()
        if eval_thread is not None:
            eval_thread.terminate()
    finally:
        if eval_thread is not None:
            eval_thread.join()


# Evaluate
def evaluate_thread(agent, step, statistics, eval_manager, log_manager):
    score, frames = eval_manager.evaluate(agent, step)
    statistics["score"] = score
    print(f"Step : {step} / {statistics}")
    log_manager.write(statistics, frames, step)


# Optimize
def optimize_thread(
    agent,
    step,
    transitions,
    interact_sync_queue,
    result_queue,
    manage_sync_queue,
    print_signal,
    save_signal,
    save_path,
    heap,
):
    opt_iter = 1
    if heap["opt_time"] is not None:
        opt_iter = max(1, heap["q_get_time"] // heap["opt_time"])
    for i in range(opt_iter):
        result = agent.process([], step) if i else agent.process(transitions, step)
        try:
            interact_sync_queue.get_nowait()
        except:
            pass
        interact_sync_queue.put(agent.sync_out())
        result_queue.put((step, result))
        if print_signal:
            try:
                manage_sync_queue.get_nowait()
            except:
                pass
            manage_sync_queue.put(agent.sync_out())
    if save_signal:
        agent.save(save_path)
