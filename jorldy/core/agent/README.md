# How to customize agent

## 1. Inherit BaseAgent class.
- If you want to add a new agent without inheriting the provided agents, you must inherit the base agent.
- Every agent must includes keward arguments when defined. You should use \*\*kwargs in \_\_init\_\_.

reference: [dqn.py](./dqn.py), [reinforce.py](./reinforce.py), [sac.py](./sac.py), ...

## 2. Implement abstract methods.
- Abstract methods(__act__, __learn__, __process__, __save__, __load__) should be implemented. Implement these methods by referring to the comments.
- When implementing a __process__, it is easy to manage events using __time_t__, __delta_t__, __event_period__, and __event_stamp__. 
  - __time_t__ means the timestamp that the agent interacted with. 
  - __delta_t__ means the difference between the new timestamp received when the __process__ is executed and the previous __time_t__. 
  - __event_period__ means the period in which the event should be executed. 
  - __event_stamp__ is added to __delta_t__ each time the __process__ is executed, and if it is greater than or equal to __event_period__, a specific event is fired.


reference: [dqn.py](./dqn.py), ...

## 3. If necessary, implement another method.
- __sync_in__, __sync_out__ methods are implemented base on __self.network__. if don't use self.network(e.g. self.actor) in agent class, should override this method.

reference: [ddpg.py](./ddpg.py), [sac.py](./sac.py), ...

- Override __set_distributed__ if you need additional work on the workers initialization.
- Override __interact_callback__ if you need additional work after interact(agent.act and env.step).

reference: [ape_x.py](./ape_x.py), ...

