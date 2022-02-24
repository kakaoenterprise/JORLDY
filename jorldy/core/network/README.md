# How to customize network 

## 1. Inherit BaseNetwork class.
- If you want to add a new network with using head in [head.py](./head.py), you must inherit the base network.

reference: [q_network.py](./q_network.py), [policy_value.py](./policy_value.py), ...

- If not, inherit torch.nn.Module.

reference: [icm.py](./icm.py), [rnd.py](./rnd.py)

## 2. If you inherit BaseNetwork, override all methods.
- __\_\_init\_\___, __forward__  methods should be overrided.
- When override __\_\_init\_\___, should consider head class. D_head_out means dimension of the embedded feature through the head.
- When override __foward__, should pass head network using super().forward(x).

reference: [q_network.py](./q_network.py), [policy_value.py](./policy_value.py), ...

