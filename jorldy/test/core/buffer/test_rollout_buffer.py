import numpy as np

from core.buffer.rollout_buffer import RolloutBuffer


def test_rollout_buffer(mock_transition):
    memory = RolloutBuffer()

    # test after init
    assert isinstance(memory.buffer, list)
    assert memory.size == 0

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
