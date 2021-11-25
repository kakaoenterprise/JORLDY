import torch

from core.network.noisy import Noisy


def test_noisy_factorized_call():
    D_in, D_out, D_hidden = 2, 3, 4
    noise_type = "factorized"
    net = Noisy(D_in=D_in, D_out=D_out, noise_type=noise_type, D_hidden=D_hidden)

    batch_size = 5
    mock_input = torch.rand((batch_size, D_in))

    out = net(mock_input, is_train=True)
    assert out.shape == (batch_size, D_out)

    out = net(mock_input, is_train=False)
    assert out.shape == (batch_size, D_out)


def test_noisy_independent_call():
    D_in, D_out, D_hidden = 2, 3, 4
    noise_type = "independent"
    net = Noisy(D_in=D_in, D_out=D_out, noise_type=noise_type, D_hidden=D_hidden)

    batch_size = 5
    mock_input = torch.rand((batch_size, D_in))

    out = net(mock_input, is_train=True)
    assert out.shape == (batch_size, D_out)

    out = net(mock_input, is_train=False)
    assert out.shape == (batch_size, D_out)
