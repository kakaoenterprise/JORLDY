# How to customize new buffer

## 1. Inherit BaseBuffer class.
- If you want to add a new buffer without inheriting the provided buffers, you must inherit the base buffer.

reference: [replay_buffer.py](./replay_buffer.py), [rollout_buffer.py](./rollout_buffer.py), ...

## 2. Implement abstract methods.
- Abstract methods(__store__, __sample__) should be implemented. Implement these methods by referring to the comments.
- When implementing __store__, it is recommended to check transition data dimension using __check_dim__. to use the __check_dim__, run __super().\_\_init\_\_()__ in the __\_\_init\_\___.

reference: [replay_buffer.py](./replay_buffer.py), [rollout_buffer.py](./rollout_buffer.py), ...
