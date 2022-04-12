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
    distributed_manager.sync(sync_queue.get(), init=True)
    step = 0
    try:
        while step < run_step:
            transitions, completed_ratio = distributed_manager.run(update_period)
            step += update_period * completed_ratio
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

    heap = {"step": 0, "run_step": run_step, "wait_thread": False, "wait_process": True}
    step = 0
    gath_thread = Thread(
        target=gather_thread,
        args=(result_queue, metric_manager, heap, "append"),
    )
    gath_thread.start()
    try:
        while step < heap["run_step"]:
            agent.sync_in(**sync_queue.get())
            while heap["wait_process"]:
                time.sleep(0.1)
            heap["wait_thread"] = True
            step = heap["step"]
            statistics = metric_manager.get_statistics()
            heap["wait_thread"] = False
            score, frames = eval_manager.evaluate(agent, step)
            statistics["score"] = score
            print(f"Step : {int(step)} / {statistics}")
            log_manager.write(statistics, frames, step)
    except Exception as e:
        traceback.print_exc()
    finally:
        gath_thread.join()


# Gather
def gather_thread(queue, target, heap, mode):
    stamp_keys = [key for key in heap.keys() if "stamp" in key]
    while heap["step"] < heap["run_step"]:
        _step, item = queue.get()
        while heap["wait_thread"]:
            time.sleep(0.1)
        heap["wait_process"] = True
        delta_t = _step - heap["step"]
        for key in stamp_keys:
            heap[key] += delta_t
        heap["step"] = _step
        if mode == "+=":
            target += item
        elif mode == "append":
            target.append(item)
        heap["wait_process"] = False
