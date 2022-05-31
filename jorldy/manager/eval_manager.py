import numpy as np
import time


class EvalManager:
    def __init__(
        self,
        Env,
        env_config,
        iteration=10,
        record=None,
        record_period=None,
        time_limit=None,
    ):
        self.env = Env(**env_config, train_mode=False)
        self.env_class = Env
        self.env_config = env_config
        self.iteration = iteration if iteration else 10
        assert iteration > 0
        self.record = record and self.env.recordable()
        self.record_period = record_period
        self.record_stamp = 0
        self.time_limit = time_limit
        self.time_t = 0

    def evaluate(self, agent, step):
        scores = []
        frames = []
        self.record_stamp += step - self.time_t
        self.time_t = step
        record = self.record and self.record_stamp >= self.record_period

        for i in range(self.iteration):
            done = False
            state = self.env.reset()
            start_time = time.time()
            while not done:
                # record first iteration
                if record and i == 0:
                    frames.append(self.env.get_frame())
                action_dict = agent.act(state, training=False)
                next_state, reward, done = self.env.step(action_dict["action"])

                # check time limit
                if (
                    self.time_limit is not None
                    and time.time() - start_time > self.time_limit
                ):
                    print(
                        f"### The evaluation time for one episode exceeded the limit. {self.time_limit} Sec ###"
                    )
                    score = self.env.score
                    self.env = self.env_class(**self.env_config, train_mode=False)
                    self.env.score = score
                    done = True

                transition = {
                    "state": state,
                    "next_state": next_state,
                    "reward": reward,
                    "done": done,
                }
                transition.update(action_dict)
                agent.interact_callback(transition)
                state = next_state
            scores.append(self.env.score)

        if record:
            self.record_stamp -= self.record_period
        return round(np.mean(scores), 4), frames
