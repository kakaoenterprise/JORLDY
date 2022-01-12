import torch

from core.network.policy_value import (
    DiscretePolicyValue,
    ContinuousPolicyValue,
    DiscretePolicySeparateValue,
    ContinuousPolicySeparateValue,
)


def test_discrete_policy_value_call():
    D_in, D_out, D_hidden = 2, 3, 4
    net = DiscretePolicyValue(D_in=D_in, D_out=D_out, D_hidden=D_hidden)

    batch_size = 5
    mock_input = torch.rand((batch_size, D_in))
    out = net(mock_input)

    assert out[0].shape == (batch_size, D_out)
    assert out[1].shape == (batch_size, 1)


def test_discrete_policy_separate_value_get_v_i():
    D_in, D_out, D_hidden = 2, 3, 4
    net = DiscretePolicySeparateValue(D_in=D_in, D_out=D_out, D_hidden=D_hidden)

    batch_size = 5
    mock_input = torch.rand((batch_size, D_in))
    out = net.get_v_i(mock_input)

    assert out.shape == (batch_size, 1)


def test_continuous_policy_value_call():
    D_in, D_out, D_hidden = 2, 3, 4
    net = ContinuousPolicyValue(D_in=D_in, D_out=D_out, D_hidden=D_hidden)

    batch_size = 5
    mock_input = torch.rand((batch_size, D_in))
    out = net(mock_input)

    assert out[0].shape == (batch_size, D_out)
    assert out[1].shape == (batch_size, D_out)
    assert out[2].shape == (batch_size, 1)


def test_continuous_policy_separate_value_get_v_i():
    D_in, D_out, D_hidden = 2, 3, 4
    net = ContinuousPolicySeparateValue(D_in=D_in, D_out=D_out, D_hidden=D_hidden)

    batch_size = 5
    mock_input = torch.rand((batch_size, D_in))
    out = net.get_v_i(mock_input)

    assert out.shape == (batch_size, 1)
