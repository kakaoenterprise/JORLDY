from core import *
from managers import *

# import config.YOUR_AGENT.YOUR_ENV as config
import config.per.assault as config

env = Env(**config.env)
agent = Agent(state_size=env.state_size,
              action_size=env.action_size,
              **config.agent)

training = config.train["training"]
load_path = config.train["load_path"]
if load_path:
    agent.load(load_path)

train_step = config.train["train_step"] if training else 0
test_step = config.train["test_step"]
print_term = config.train["print_term"]
save_term = config.train["save_term"]

test_manager = TestManager()
metric_manager = MetricManager()
log_id = config.agent["name"] if "id" not in config.train.keys() else config.train["id"]
purpose = None if "purpose" not in config.train.keys() else config.train["purpose"]
log_manager = LogManager(config.env["name"], log_id, purpose)

episode = 0
state = env.reset()
for step in range(train_step + test_step):
    if step == train_step:
        if training:
            agent.save(log_manager.path)
            training = False
        print("### TEST START ###")
    
    action = agent.act(state, training)
    next_state, reward, done = env.step(action)
    
    if training:
        result = agent.process(state, action, reward, next_state, done)
        if result:
            metric_manager.append(result)
    state = next_state
    
    if done:
        episode += 1

        mode = "train" if training else "test"
        metric_manager.append({f"{mode}_score": env.score})
            
        if episode % print_term == 0:
            if training:
                score = test_manager.test(agent, env, config.train["test_iteration"])
                metric_manager.append({"test_score": score})
            statistics = metric_manager.get_statistics()
            print(f"{episode} Episode / Step : {step} / {statistics}")
            log_manager.write_scalar(statistics, step)
        
        if training and episode % save_term == 0:
            agent.save(log_manager.path)
        
        state = env.reset()
        
env.close()
