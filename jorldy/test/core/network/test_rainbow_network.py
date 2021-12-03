import torch

from core.network.rainbow import Rainbow


def test_rainbow_call():
    D_in, D_out, D_hidden = 2, 3, 4
    N_atom = 5
    noise_type = "factorized"

    net = Rainbow(
        D_in=D_in, D_out=D_out, N_atom=N_atom, noise_type=noise_type, D_hidden=D_hidden
    )

    batch_size = 6
    mock_input = torch.rand((batch_size, D_in))

    out = net(mock_input, is_train=True)
    assert out.shape == (batch_size, D_out, N_atom)
