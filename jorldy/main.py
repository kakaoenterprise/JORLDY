import argparse

from run_mode import *

# default_config_path = "config.YOUR_AGENT.YOUR_ENV"
default_config_path = "config.dqn.cartpole"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--async", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--config", type=str, help="config.dqn.cartpole")
    args, unknown = parser.parse_known_args()

    choosed_mode_num = args.single + args.sync + args.__dict__["async"] + args.eval
    assert choosed_mode_num < 2, "You have to choose only one mode"

    config_path = args.config if args.config else default_config_path

    if args.single or choosed_mode_num == 0:
        single_train(config_path, unknown)
    elif args.sync:
        sync_distributed_train(config_path, unknown)
    elif args.__dict__["async"]:
        async_distributed_train(config_path, unknown)
    elif args.eval:
        evaluate(config_path, unknown)
