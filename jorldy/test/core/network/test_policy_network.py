import torch

from core.network.policy import DeterministicPolicy, DiscretePolicy, ContinuousPolicy


def test_deterministic_policy_call():
    D_in, D_out, D_hidden = 2, 3, 4
    net = DeterministicPolicy(D_in=D_in, D_out=D_out, D_hidden=D_hidden)

    batch_size = 5
    mock_input = torch.rand((batch_size, D_in))
    out = net(mock_input)

    assert out.shape == (batch_size, D_out)


def test_discrete_policy_call():
    D_in, D_out, D_hidden = 2, 3, 4
    net = DiscretePolicy(D_in=D_in, D_out=D_out, D_hidden=D_hidden)

    batch_size = 5
    mock_input = torch.rand((batch_size, D_in))
    out = net(mock_input)

    assert out.shape == (batch_size, D_out)


def test_continuous_policy_call():
    D_in, D_out, D_hidden = 2, 3, 4
    net = ContinuousPolicy(D_in=D_in, D_out=D_out, D_hidden=D_hidden)

    batch_size = 5
    mock_input = torch.rand((batch_size, D_in))
    out = net(mock_input)

    assert out[0].shape == (batch_size, D_out)
    assert out[1].shape == (batch_size, D_out)
