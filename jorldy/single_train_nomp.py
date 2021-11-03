import argparse

from core import *
from manager import *

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
    agent = Agent(state_size=env.state_size,
                  action_size=env.action_size,
                  optim_config=config.optim,
                  **config.agent)

    if config.train.load_path:
        agent.load(config.train.load_path)

    record_period = config.train.record_period if config.train.record_period else config.train.run_step//10
    eval_manager = EvalManager(Env(**config.env), config.train.eval_iteration, 
                               config.train.record, record_period)
    metric_manager = MetricManager()
    log_id = config.train.id if config.train.id else config.agent.name
    log_manager = LogManager(config.env.name, log_id, config.train.experiment)
    config_manager.dump(log_manager.path)
    
    episode = 0
    state = env.reset()

    for step in range(1, config.train.run_step+1):
        action_dict = agent.act(state, config.train.training)            
        next_state, reward, done = env.step(action_dict['action'])

        if config.train.training:
            transition = {'state': state, 'next_state': next_state,
                          'reward': reward, 'done': done}
            transition.update(action_dict)
            transition = agent.interact_callback(transition)
            if transition:
                result = agent.process([transition], step)
                metric_manager.append(result)
        state = next_state

        if done:
            episode += 1
            state = env.reset()

        if step % config.train.print_period == 0:
            score, frames = eval_manager.evaluate(agent, step)
            metric_manager.append({"score": score})
            statistics = metric_manager.get_statistics()
            print(f"{episode} Episode / Step : {step} / {statistics}")
            log_manager.write(statistics, frames, score, step)

        if config.train.training and \
            (step % config.train.save_period == 0 or step == config.train.run_step - 1):
            agent.save(log_manager.path)
        
    env.close()
