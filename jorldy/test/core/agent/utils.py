import os, shutil
import numpy as np
import torch


class MockEnv:
    def __init__(self, state_size, action_size, action_type, episode_len):
        self.state_size = state_size
        self.action_size = action_size
        self.action_type = action_type
        self.episode_len = episode_len
        self.time_t = 0

    def reset(self):
        state = np.random.random((1, self.state_size))
        return state

    def step(self, action):
        self.time_t += 1
        next_state = np.random.random((1, self.state_size))
        reward = np.random.random((1, 1))
        done = np.array([[self.time_t == self.episode_len]])
        if done:
            self.time_t = 0
        return next_state, reward, done


def check_interact(env, agent, run_step):
    state = env.reset()
    for step in range(1, run_step + 1):
        action_dict = agent.act(state)
        action_branch = 1 if agent.action_type == "discrete" else env.action_size
        assert action_dict["action"].shape == (1, action_branch)

        next_state, reward, done = env.step(action_dict["action"])
        transition = {
            "state": state,
            "next_state": next_state,
            "reward": reward,
            "done": done,
        }
        transition.update(action_dict)
        transition = agent.interact_callback(transition)
        if transition:
            result = agent.process([transition], step)
        state = next_state


def check_save_load(agent, path):
    try:
        os.makedirs(path, exist_ok=True)
        agent.save(path)
        agent.load(path)
    except Exception as e:
        raise e
    finally:
        shutil.rmtree(path)


def check_sync_in_out(agent):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.sync_in(**agent.sync_out(device))
