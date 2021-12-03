import os, shutil
import torch


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
