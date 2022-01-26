import torch

from core.network.rnd import RND_MLP, RND_CNN, RND_Multi


def test_rnd_mlp_call():
    D_in, D_out, D_hidden = 2, 3, 4
    num_workers, gamma_i = 2, 0.99
    net = RND_MLP(
        D_in=D_in,
        D_out=D_out,
        D_hidden=D_hidden,
        num_workers=num_workers,
        gamma_i=gamma_i,
    )

    batch_size = 5
    mock_input = [
        torch.rand((batch_size * num_workers, D_in)),
    ]
    out = net(*mock_input, update_ri=True)

    assert out.shape == (batch_size * num_workers, 1)

    mock_input = [
        torch.rand((batch_size, D_in)),
    ]
    out = net(*mock_input, update_ri=False)

    assert out.shape == (batch_size, 1)


def test_rnd_cnn_call():
    D_in, D_out, D_hidden = [3, 36, 36], 3, 4
    num_workers, gamma_i = 2, 0.99
    net = RND_CNN(
        D_in=D_in,
        D_out=D_out,
        D_hidden=D_hidden,
        num_workers=num_workers,
        gamma_i=gamma_i,
    )

    batch_size = 5
    mock_input = [
        torch.rand((batch_size * num_workers, *D_in)),
    ]
    out = net(*mock_input, update_ri=True)

    assert out.shape == (batch_size * num_workers, 1)

    mock_input = [
        torch.rand((batch_size, *D_in)),
    ]
    out = net(*mock_input, update_ri=False)

    assert out.shape == (batch_size, 1)


def test_rnd_multi_call():
    D_in, D_out, D_hidden = [[3, 36, 36], 2], 3, 4
    num_workers, gamma_i = 2, 0.99
    net = RND_Multi(
        D_in=D_in,
        D_out=D_out,
        D_hidden=D_hidden,
        num_workers=num_workers,
        gamma_i=gamma_i,
    )

    batch_size = 5
    mock_input = [
        [
            torch.rand((batch_size * num_workers, *D_in[0])),
            torch.rand((batch_size * num_workers, D_in[1])),
        ],
    ]
    out = net(*mock_input, update_ri=True)

    assert out.shape == (batch_size * num_workers, 1)

    mock_input = [
        [torch.rand((batch_size, *D_in[0])), torch.rand((batch_size, D_in[1]))],
    ]
    out = net(*mock_input, update_ri=False)

    assert out.shape == (batch_size, 1)
