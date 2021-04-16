from core import *
from managers import *

# import config.YOUR_AGENT.YOUR_ENV as config
import config.ppo.pong_mlagent as config

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

test_manager = TestManager()
metric_manager = MetricManager()
log_id = config.agent["name"] if "id" not in config.train.keys() else config.train["id"]
log_manager = LogManager(config.env["name"], log_id)
distributed_manager = DistributedManager(Env, config.env, agent, config.train["num_worker"])

step = 0
while step < run_step:
    step += update_term
    transitions = distributed_manager.run(update_term)
    result = agent.process(transitions)
    
    if result:
        metric_manager.append(result)
        distributed_manager.sync(agent)
    
    if step % print_term == 0:
        score = test_manager.test(agent, env, config.train["test_iteration"])
        metric_manager.append({"score": score})
        statistics = metric_manager.get_statistics()
        print(f"Step : {step} / {statistics}")
        log_manager.write_scalar(statistics, step)

    if step % save_term == 0 or step == run_step:
        agent.save(log_manager.path)
        
env.close()