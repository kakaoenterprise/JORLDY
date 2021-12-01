import torch

from core.network.dueling import Dueling


def test_dueling_call():
    D_in, D_out, D_hidden = 2, 3, 4
    net = Dueling(D_in=D_in, D_out=D_out, D_hidden=D_hidden)

    batch_size = 5
    mock_input = torch.rand((batch_size, D_in))
    out = net(mock_input)

    assert out.shape == (batch_size, D_out)
