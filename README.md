# JORLDY (Beta)

[![license badge](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

Hello Wo**RL**d!!:hand:  **Join Our Reinforcement Learning framework for Developing Yours (JORLDY)** is an open-source Reinforcement Learning (RL) framework provided by [KakaoEnterprise](https://www.kakaoenterprise.com/). It is named after Jordy, one of the [Kakao Niniz](https://www.kakaocorp.com/page/service/service/Niniz) character. It provides various RL algorithms and environment and they can be easily used using single code. This repository is opened for helping RL researchers and students who study RL.


## :fire: Features

- 20+ RL Algorithms and various RL environment are provided
- Algorithms and environment are customizable
- New algorithms and environment can be added 
- Distributed RL algorithms are provided using [ray](https://github.com/ray-project/ray)
- Benchmark of the algorithms is conducted in many RL environment

## :heavy_check_mark: Tested

| Python |   Windows   |   Mac   |   Linux  |
| :----: | :---------: | :-----: | :------: |
|  3.8<br />3.7  | Windows Server 2022 | macOS Big Sur 11<br />macOS Catalina 10.15 | Ubuntu 20.04<br />Ubuntu 18.04 |

## :arrow_down: Installation

```
git clone https://github.com/kakaoenterprise/JORLDY.git  
cd JORLDY
pip install -r requirements.txt

# linux
apt-get update 
apt-get -y install libgl1-mesa-glx # for opencv
apt-get -y install libglib2.0-0    # for opencv
apt-get -y install gifsicle        # for gif optimize
```

### :whale: To use docker

(customize if necessary)
```
cd JORLDY

docker pull jorldy/jorldy

# mac, linux
docker run -it --rm --name jorldy -v `pwd`:/JORLDY jorldy/jorldy /bin/bash

# windows
docker run -it --rm --name jorldy -v %cd%:/JORLDY jorldy/jorldy /bin/bash
```

### :heavy_plus_sign: To use additional environments

__- Atari and Super Mario Bros__

atari and super-mario-bros need to be installed manually due to licensing issues

```
# To use atari
pip install --upgrade gym[atari,accept-rom-license]
 
# To use super-mario-bros
pip install gym-super-mario-bros
```

__- Mujoco__ (Mac and Linux only)

__Mujoco is supported in docker.__  However, if you don't use docker, several subprocesses should be done.
Please refer to the [mujoco-py github installation](https://github.com/openai/mujoco-py#install-mujoco)


## :rocket: Getting started

```
cd jorldy

# Examples: python main.py [run mode] --config [config path]
python main.py --config config.dqn.cartpole
python main.py --async --config config.ape_x.cartpole

# Examples: python main.py [run mode] --config [config path] --[optional parameter key] [parameter value]
python main.py --config config.rainbow.atari --env.name breakout
python main.py --sync --config config.ppo.cartpole --train.num_workers 8

```

## :card_index_dividers: Release 

| Version |   Release Date   |   Source   |   Release Note  |
| :-----: | :--------------: | :--------: | :----------: |
|  0.2.0  | January 23, 2022 | [Source](https://github.com/kakaoenterprise/JORLDY/tree/v0.2.0) | [Release Note](https://github.com/kakaoenterprise/JORLDY/releases/tag/v0.2.0) |
|  0.1.0  | December 23, 2021 | [Source](https://github.com/kakaoenterprise/JORLDY/tree/v0.1.0) | [Release Note](https://github.com/kakaoenterprise/JORLDY/releases/tag/v0.1.0) |
|  0.0.3  | November 23, 2021 | [Source](https://github.com/kakaoenterprise/JORLDY/tree/v0.0.3) | [Release Note](https://github.com/kakaoenterprise/JORLDY/releases/tag/v0.0.3) |
|  0.0.2  | November 06, 2021 | [Source](https://github.com/kakaoenterprise/JORLDY/tree/v0.0.2) | [Release Note](https://github.com/kakaoenterprise/JORLDY/releases/tag/v0.0.2) |
|  0.0.1  | November 03, 2021 | [Source](https://github.com/kakaoenterprise/JORLDY/tree/v0.0.1) | [Release Note](https://github.com/kakaoenterprise/JORLDY/releases/tag/v0.0.1) |


## :mag: How to

- [How to use](./docs/How_to_use.md)
- [How to customize config](./jorldy/config/README.md)
- [How to customize agent](./jorldy/core/agent/README.md)
- [How to customize environment](./jorldy/core/env/README.md)
- [How to customize network](./jorldy/core/network/README.md)
- [How to customize buffer](./jorldy/core/buffer/README.md)


## :page_facing_up: Documentation

- [Distributed Architecture](./docs/Distributed_Architecture.md)
- [Role of Managers](./jorldy/manager/README.md)
- [List of Contents](./docs/List_of_Contents.md)
- [Naming Convention](./docs/Naming_convention.md)
- [Benchmark](https://petite-balance-8cb.notion.site/Benchmark-09684f1adf764c84a5a331cb5690544f)
- [Reference](./docs/Reference.md)


## :busts_in_silhouette: Contributors

:mailbox: Contact: jorldy@kakaoenterprise.com

<img src="./resrc/contributors.png" alt="contributors" width=80%/> 


## :copyright: License

[Apache License 2.0](./LICENSE.md)

## :no_entry_sign: Disclaimer

Installing in JORLDY and/or utilizing algorithms or environments not provided KEP may involve a use of third party’s intellectual property. It is advisable that a user obtain licenses or permissions from the right holder(s), if necessary, or take any other necessary measures to avoid infringement or misappropriation of third party’s intellectual property rights.
