import torch

from core.network.dqn import DQN


def test_dqn_call():
    D_in, D_out, D_hidden = 2, 3, 4
    net = DQN(D_in=D_in, D_out=D_out, D_hidden=D_hidden)

    batch_size = 5
    mock_input = torch.rand((batch_size, D_in))
    out = net(mock_input)

    assert out.shape == (batch_size, D_out)
