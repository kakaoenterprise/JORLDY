import numpy as np
from core.buffer.rollout_buffer import RolloutBuffer


def test_rollout_buffer():
    memory = RolloutBuffer()

    # test after init
    assert isinstance(memory.buffer, list)
    assert memory.size == 0

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

    # test store
    store_iteration = 15
    for _ in range(store_iteration):
        memory.store(mock_transition)

    # test after store
    assert memory.size == store_iteration

    # test sample
    sample_transitions = memory.sample()
    assert isinstance(sample_transitions, dict)
    for key, val in sample_transitions.items():
        assert key in mock_transition[0].keys()
        if isinstance(val, list):
            for i, v in enumerate(val):
                assert isinstance(v, np.ndarray)
                assert v.shape == (
                    store_iteration,
                    *mock_transition[0][key][i].shape[1:],
                )
        else:
            assert isinstance(val, np.ndarray)
            assert val.shape == (store_iteration, *mock_transition[0][key].shape[1:])

    # test after sample
    assert memory.size == 0
