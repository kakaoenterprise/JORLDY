import numpy as np

mock_transition = [
    {
        "state": np.random.random((1, 4)),
        "action": np.random.random((1, 3)),
        "reward": np.random.random((1, 1)),
        "next_state": np.random.random((1, 4)),
        "done": np.random.random((1, 1)) < 0.5,
        "multi_modal": [np.random.random((1, 3, 8, 8)), np.random.random((1, 4))],
        "seq": np.random.random((1, 3, 4)),
    },
]
