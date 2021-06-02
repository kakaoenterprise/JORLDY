from core import *
from managers import *

# import config.YOUR_AGENT.YOUR_ENV as config
import config.icm_ppo.cartpole as config

if __name__=="__main__":
    env = Env(**config.env)
    agent = Agent(state_size=env.state_size,
                  action_size=env.action_size,
                  **config.agent)

    training = config.train["training"]
    load_path = config.train["load_path"]
    if load_path:
        agent.load(load_path)

    run_step = config.train["run_step"]
    print_period = config.train["print_period"]
    save_period = config.train["save_period"]

    test_manager = TestManager(config.train["test_iteration"])
    metric_manager = MetricManager()
    time_manager = TimeManager()
    log_id = config.agent["name"] if "id" not in config.train.keys() else config.train["id"]
    purpose = None if "purpose" not in config.train.keys() else config.train["purpose"]
    log_manager = LogManager(config.env["name"], log_id, purpose)

    episode = 0
    state = env.reset()
    for step in range(1, run_step+1):
        action = agent.act(state, training)
        next_state, reward, done = env.step(action)

        if training:
            result = agent.process([(state, action, reward, next_state, done)], step)
            if result:
                metric_manager.append(result)
        state = next_state

        if done:
            episode += 1
            state = env.reset()

        if step % print_period == 0:
            score = test_manager.test(agent, env)
            metric_manager.append({"score": score})
            statistics = metric_manager.get_statistics()
            print(f"{episode} Episode / Step : {step} / {statistics}")
            log_manager.write_scalar(statistics, step)

        if training and (step % save_period == 0 or step == run_step - 1):
            agent.save(log_manager.path)


    env.close()
