import argparse

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

    record_period = config.train.record_period if config.train.record_period else config.train.run_step//10

    eval_manager = EvalManager(Env(**config.env), config.train.eval_iteration, config.train.record, record_period)
    metric_manager = MetricManager()
    log_id = config.train.id if config.train.id else config.agent.name
    log_manager = LogManager(config.env.name, log_id, config.train.experiment)
    config_manager.dump(log_manager.path)

    distributed_manager = DistributedManager(Env, config.env, Agent, {'device':'cpu', **agent_config}, config.train.num_workers, 'sync')
    
    step, print_stamp, save_stamp = 0, 0, 0
    while step < config.train.run_step:
        transitions = distributed_manager.run(config.train.update_period)
        step += config.train.update_period
        print_stamp += config.train.update_period
        save_stamp += config.train.update_period
        result = agent.process(transitions, step)
        distributed_manager.sync(agent.sync_out())
        metric_manager.append(result)
        if print_stamp >= config.train.print_period or step >= config.train.run_step:
            score, frames = eval_manager.evaluate(agent, step)
            metric_manager.append({"score": score})
            statistics = metric_manager.get_statistics()
            print(f"Step : {step} / {statistics}")
            log_manager.write(statistics, frames, score, step)
            print_stamp = 0
        if save_stamp >= config.train.save_period or step >= config.train.run_step:
            agent.save(log_manager.path)
            save_stamp = 0
    env.close()