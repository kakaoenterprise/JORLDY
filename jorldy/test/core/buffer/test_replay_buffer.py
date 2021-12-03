import numpy as np

from core.buffer.replay_buffer import ReplayBuffer


def test_replay_buffer(mock_transition):
    buffer_size = 10
    memory = ReplayBuffer(buffer_size=buffer_size)

    # test after init
    assert memory.buffer_size == buffer_size
    assert memory.buffer_index == 0
    assert memory.size == 0

    # test store
    store_iteration = 15
    for _ in range(store_iteration):
        memory.store(mock_transition)

    # test after store
    assert memory.buffer_index == (store_iteration % buffer_size)
    assert memory.size == min(buffer_size, store_iteration)

    # test sample
    batch_size = 8
    sample_transitions = memory.sample(batch_size=batch_size)
    assert isinstance(sample_transitions, dict)
    for key, val in sample_transitions.items():
        assert key in mock_transition[0].keys()
        if isinstance(val, list):
            for i, v in enumerate(val):
                assert isinstance(v, np.ndarray)
                assert v.shape == (batch_size, *mock_transition[0][key][i].shape[1:])
        else:
            assert isinstance(val, np.ndarray)
            assert val.shape == (batch_size, *mock_transition[0][key].shape[1:])
