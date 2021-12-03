import numpy as np

from core.buffer.per_buffer import PERBuffer


def test_per_buffer(mock_transition):
    buffer_size = 10
    uniform_sample_prob = 1e-3
    memory = PERBuffer(buffer_size=buffer_size, uniform_sample_prob=uniform_sample_prob)

    # test after init
    assert memory.buffer_size == buffer_size
    assert memory.tree_size == (buffer_size * 2) - 1
    assert memory.buffer_index == 0
    assert memory.tree_index == buffer_size - 1
    assert memory.size == 0

    # test store
    store_iteration = 15
    for _ in range(store_iteration):
        memory.store(mock_transition)

    # test after store
    assert memory.buffer_index == (store_iteration % buffer_size)
    assert memory.tree_index == buffer_size - 1 + (store_iteration % buffer_size)
    assert memory.size == min(buffer_size, store_iteration)

    # test sample
    beta, batch_size = 0.4, 8
    sample_transitions, weiights, indices, sampled_p, mean_p = memory.sample(
        beta=beta, batch_size=batch_size
    )
    assert isinstance(sample_transitions, dict)
    assert isinstance(weiights, np.ndarray)
    assert (weiights <= 1.0).all()
    assert isinstance(indices, np.ndarray)
    assert (indices >= buffer_size - 1).all()
    assert isinstance(sampled_p, float)
    assert isinstance(mean_p, float)

    for key, val in sample_transitions.items():
        assert key in mock_transition[0].keys()
        if isinstance(val, list):
            for i, v in enumerate(val):
                assert isinstance(v, np.ndarray)
                assert v.shape == (batch_size, *mock_transition[0][key][i].shape[1:])
        else:
            assert isinstance(val, np.ndarray)
            assert val.shape == (batch_size, *mock_transition[0][key].shape[1:])

    # test update_priority
    new_priority, index = 2.0, buffer_size - 1 + (buffer_size // 2)
    memory.update_priority(new_priority, index)

    assert memory.max_priority == new_priority
    assert memory.sum_tree[buffer_size - 1 + (buffer_size // 2)] == new_priority
