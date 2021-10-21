# Class naming convention
- We basically follow the PascalCase naming conventions in which the first letter of each word in a compound word is capitalized. 
    - Example: DiscretePolicy, MountainCar, ...
- Write all acronyms in uppercase.
    - Example: DQN, CNN, SAC, PPO, PongMLAgent
- Acronyms and words (including abbreviations) are separated by '_'.
    - Example: ICM_PPO, SAC_Critic, ...
- Exeptional case:
    - CartPole -> Cartpole
        - The exception rule is applied to speed up debugging.
    - SuperMarioBros -> Mario
        - The exception rule is applied because the class name is too long.

# Class calling convention
- If there are consecutive lowercase and uppercase letters, add '_' between them.
- Change all uppercase letters to lowercase 
- Example: 
    - DiscretePolicyValue -> discrete_policy_value
    - SAC_Critic -> sac_critic
    - PongMLAgent -> pong_mlagent

# Function, Variable naming convention
- We follow the Snake case in which each space is replaced by an underscore (_) character, and the first letter of each word written in lowercase.