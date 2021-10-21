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
 $ git clone https://github.com/kakaoenterprise/jorldy.git  
 $ cd jorldy
 $ pip install -r requirements.txt  
 $ pip install gym[atari]
 $ pip install gym-super-mario-bros
```



## :rocket: QuickStart

<img src="./resrc/quickstart.png" alt="quickstart" width=60%/> 



## :card_index_dividers: Release 

| Version |   Release Date   |   Source   |   Download   |
| :-----: | :--------------: | :--------: | :----------: |
|  1.0.0  | October xx, 2021 | [Source]() | [Download]() |



## :mag: How to

- [How to use](./docs/How_to_use.md)
- [How to customize config](./config/README.md)
- [How to customize agent](./core/agent/README.md)
- [How to customize environment](./core/env/README.md)
- [How to customize network](./core/network/README.md)
- [How to customize buffer](./core/buffer/README.md)



## :page_facing_up: Documentation

- [Distributed Architecture](./docs/Distributed_Architecture.md)
- [Role of Managers](./manager/README.md)
- [Implementation List](./docs/Implementation_list.md)
- [Naming Convention](./docs/Naming_convention.md)
- [Benchmark](https://www.notion.so/rlnote/Benchmark-c7642d152cad4980bc03fe804fe9e88a)
- [Reference](./docs/Reference.md)



## :busts_in_silhouette: Contributors

:mailbox: Contact: atech.rl@kakaocorp.com

<img src="./resrc/contributors.png" alt="contributors" width=80%/> 


## :copyright: License

[Apache License 2.0]()



