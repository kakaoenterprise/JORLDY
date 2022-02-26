import os
import datetime, time

import imageio
from pygifsicle import optimize
from torch.utils.tensorboard import SummaryWriter


class LogManager:
    def __init__(self, env, id, experiment=None):
        self.id = id
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        self.path = (
            f"./logs/{experiment}/{env}/{id}/{now}/"
            if experiment
            else f"./logs/{env}/{id}/{now}/"
        )
        self.writer = SummaryWriter(self.path)
        self.stamp = time.time()

    def write(self, scalar_dict, frames, step):
        for key, value in scalar_dict.items():
            self.writer.add_scalar(f"{self.id}/" + key, value, step)
            self.writer.add_scalar("all/" + key, value, step)
            if "score" in key:
                time_delta = int(time.time() - self.stamp)
                self.writer.add_scalar(f"{self.id}/{key}_per_time", value, time_delta)
                self.writer.add_scalar(f"all/{key}_per_time", value, time_delta)

        if len(frames) > 0:
            score = scalar_dict["score"]
            write_path = os.path.join(self.path, f"{int(step):010d}_{score}.gif")
            imageio.mimwrite(write_path, frames, fps=60)
            optimize(write_path)
            print(f"...Record episode to {write_path}...")
