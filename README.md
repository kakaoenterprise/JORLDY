# JORLDY 

[![license badge](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

**Join Our Reinforcement Learning framework for Developing Yours (JORLDY)** is an open-source Reinforcement Learning (RL) framework provided by [KakaoEnterpise](https://www.kakaoenterprise.com/). It provides various RL algorithms and environment and they can be easily used using single code. This repository is opened for helping RL researchers and students who study RL.



## :fire: Features

- 20+ RL Algorithms and various RL environment are provided
- Algorithms and environment are customizable
- New algorithms are environment can be added 
- Distributed RL algorithms are provided using [ray](https://github.com/ray-project/ray)
- Benchmark of the algorithms is conducted in many RL environment


## :arrow_down: Installation

```
 $ git clone https://github.com/kakaoenterprise/JORLDY.git  
 $ cd JORLDY
 $ pip install -r requirements.txt
```

__:whale: To use docker__

(customize if necessary)
```
 $ cd JORLDY

 # build
 $ docker build -t jorldy -f ./docker/Dockerfile .
 
 # run (mac, linux)
 $ docker run -it --rm --name jorldy -v `pwd`/jorldy:/jorldy jorldy /bin/bash
```

__:heavy_plus_sign: To use additional environments__

(atari and super-mario-bros need to be installed manually due to licensing issues)
```
 # To use atari
 $ pip install --upgrade gym[atari,accept-rom-license]
 
 # To use super-mario-bros
 $ pip install gym-super-mario-bros
```

## :rocket: QuickStart

<img src="./resrc/quickstart.png" alt="quickstart" width=60%/> 



## :card_index_dividers: Release 

| Version |   Release Date   |   Source   |   Download   |
| :-----: | :--------------: | :--------: | :----------: |
|  0.0.1  | November 03, 2021 | [Source]() | [Download]() |



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
- [Benchmark](https://www.notion.so/rlnote/Benchmark-c7642d152cad4980bc03fe804fe9e88a)
- [Reference](./docs/Reference.md)



## :busts_in_silhouette: Contributors

:mailbox: Contact: atech.rl@kakaocorp.com

<img src="./resrc/contributors.png" alt="contributors" width=80%/> 


## :copyright: License

[Apache License 2.0](./LICENSE.md)

## :no_entry_sign: Disclaimer

Installing in JORDY and/or utilizing algorithms or environments not provided KEP may involve a use of third party’s intellectual property. It is advisable that a user obtain licenses or permissions from the right holder(s), if necessary, or take any other necessary measures to avoid infringement or misappropriation of third party’s intellectual property rights.