from core import *
from managers import *

# import config.YOUR_AGENT.YOUR_ENV as config
import config.dqn.pong_mlagent as config

env = Env(**config.env)
agent = Agent(state_size=env.state_size,
              action_size=env.action_size,
              **config.agent)

training = config.train["training"]
load_path = config.train["load_path"]
if load_path:
    agent.load(load_path)

run_step = config.train["run_step"]
print_term = config.train["print_term"]
save_term = config.train["save_term"]

test_manager = TestManager()
metric_manager = MetricManager()
log_id = config.agent["name"] if "id" not in config.train.keys() else config.train["id"]
log_manager = LogManager(config.env["name"], log_id)

episode = 0
state = env.reset()
for step in range(run_step):
    action = agent.act(state, training)
    next_state, reward, done = env.step(action)
    
    if training:
        result = agent.process([(state, action, reward, next_state, done)])
        if result:
            metric_manager.append(result)
    state = next_state
    
    if done:
        episode += 1
        state = env.reset()
            
    if step % print_term == 0:
        score = test_manager.test(agent, env, config.train["test_iteration"])
        metric_manager.append({"score": score})
        statistics = metric_manager.get_statistics()
        print(f"{episode} Episode / Step : {step} / {statistics}")
        log_manager.write_scalar(statistics, step)

    if training and (step % save_term == 0 or step == run_step - 1):
        agent.save(log_manager.path)
        
        
env.close()
