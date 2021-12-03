import torch

from core.network.head import MLP, CNN, Multi, MLP_LSTM, CNN_LSTM


def test_mlp_call():
    D_in, D_hidden = 2, 3
    net = MLP(D_in=D_in, D_hidden=D_hidden)

    batch_size = 4
    mock_input = torch.rand((batch_size, D_in))
    hidden = net(mock_input)

    assert hidden.shape == (batch_size, net.D_head_out)


def test_cnn_call():
    D_channel, D_height, D_width = 3, 36, 36
    D_in = [D_channel, D_height, D_width]
    net = CNN(D_in=D_in)

    batch_size = 4
    mock_input = torch.rand((batch_size, *D_in))
    hidden = net(mock_input)

    assert hidden.shape == (batch_size, net.D_head_out)


def test_multi_call():
    D_channel, D_height, D_width = 3, 36, 36
    D_in_img = [D_channel, D_height, D_width]
    D_in_vec = 2
    D_in, D_hidden = [D_in_img, D_in_vec], 3
    net = Multi(D_in=D_in, D_hidden=D_hidden)

    batch_size = 4
    mock_input = [
        torch.rand((batch_size, *D_in_img)),
        torch.rand((batch_size, D_in_vec)),
    ]
    hidden = net(mock_input)

    assert hidden.shape == (batch_size, net.D_head_out)


def test_mlp_lstm_call():
    D_in, D_hidden = 2, 3
    net = MLP_LSTM(D_in=D_in, D_hidden=D_hidden)

    batch_size, seq_len = 4, 5
    mock_input = torch.rand((batch_size, seq_len, D_in))
    x, hidden_in, hidden_out = net(mock_input)

    assert x.shape == (batch_size, seq_len, net.D_head_out)
    assert hidden_in[0].shape == (1, batch_size, D_hidden)
    assert hidden_in[1].shape == (1, batch_size, D_hidden)
    assert hidden_out[0].shape == (1, batch_size, D_hidden)
    assert hidden_out[1].shape == (1, batch_size, D_hidden)


def test_cnn_lstm_call():
    D_channel, D_height, D_width = 3, 36, 36
    D_in, D_hidden = [D_channel, D_height, D_width], 2
    net = CNN_LSTM(D_in=D_in, D_hidden=D_hidden)

    batch_size, seq_len = 3, 4
    mock_input = torch.rand((batch_size, seq_len, *D_in))
    x, hidden_in, hidden_out = net(mock_input)

    assert x.shape == (batch_size, seq_len, net.D_head_out)
    assert hidden_in[0].shape == (1, batch_size, D_hidden)
    assert hidden_in[1].shape == (1, batch_size, D_hidden)
    assert hidden_out[0].shape == (1, batch_size, D_hidden)
    assert hidden_out[1].shape == (1, batch_size, D_hidden)
