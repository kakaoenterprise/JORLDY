import torch

from core.network.r2d2 import R2D2


def test_r2d2_call():
    D_in, D_out, D_hidden = 2, 3, 4
    net = R2D2(D_in=D_in, D_out=D_out, D_hidden=D_hidden)

    batch_size = 7
    seq_len = 5

    mock_input = torch.rand((batch_size, seq_len, D_in))
    mock_prev_action = torch.zeros((batch_size, seq_len, D_out))
    mock_hidden = None

    out, hidden_in, hidden_out = net(mock_input, mock_prev_action, mock_hidden)

    assert out.shape == (batch_size, seq_len, D_out)

    assert hidden_in[0].shape == (1, batch_size, D_hidden)
    assert hidden_in[1].shape == (1, batch_size, D_hidden)

    assert hidden_out[0].shape == (1, batch_size, D_hidden)
    assert hidden_out[0].shape == (1, batch_size, D_hidden)
