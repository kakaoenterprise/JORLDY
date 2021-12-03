import torch

from core.network.icm import ICM_MLP, ICM_CNN, ICM_Multi


def test_discrete_icm_mlp_call():
    D_in, D_out, D_hidden = 2, 3, 4
    num_workers, gamma, eta = 2, 0.99, 0.1
    action_type = "discrete"
    net = ICM_MLP(
        D_in=D_in,
        D_out=D_out,
        D_hidden=D_hidden,
        num_workers=num_workers,
        gamma=gamma,
        eta=eta,
        action_type=action_type,
    )

    batch_size = 5
    mock_input = [
        torch.rand((batch_size * num_workers, D_in)),
        torch.rand((batch_size * num_workers, 1)),
        torch.rand((batch_size * num_workers, D_in)),
    ]
    out = net(*mock_input, update_ri=True)

    assert out[0].shape == (batch_size * num_workers,)
    assert out[1].shape == ()
    assert out[2].shape == ()

    mock_input = [
        torch.rand((batch_size, D_in)),
        torch.rand((batch_size, 1)),
        torch.rand((batch_size, D_in)),
    ]
    out = net(*mock_input, update_ri=False)

    assert out[0].shape == (batch_size,)
    assert out[1].shape == ()
    assert out[2].shape == ()


def test_continuous_icm_mlp_call():
    D_in, D_out, D_hidden = 2, 3, 4
    num_workers, gamma, eta = 2, 0.99, 0.1
    action_type = "continuous"
    net = ICM_MLP(
        D_in=D_in,
        D_out=D_out,
        D_hidden=D_hidden,
        num_workers=num_workers,
        gamma=gamma,
        eta=eta,
        action_type=action_type,
    )

    batch_size = 5
    mock_input = [
        torch.rand((batch_size * num_workers, D_in)),
        torch.rand((batch_size * num_workers, D_out)),
        torch.rand((batch_size * num_workers, D_in)),
    ]
    out = net(*mock_input, update_ri=True)

    assert out[0].shape == (batch_size * num_workers,)
    assert out[1].shape == ()
    assert out[2].shape == ()

    mock_input = [
        torch.rand((batch_size, D_in)),
        torch.rand((batch_size, D_out)),
        torch.rand((batch_size, D_in)),
    ]
    out = net(*mock_input, update_ri=False)

    assert out[0].shape == (batch_size,)
    assert out[1].shape == ()
    assert out[2].shape == ()


def test_discrete_icm_cnn_call():
    D_in, D_out, D_hidden = [3, 36, 36], 3, 4
    num_workers, gamma, eta = 2, 0.99, 0.1
    action_type = "discrete"
    net = ICM_CNN(
        D_in=D_in,
        D_out=D_out,
        D_hidden=D_hidden,
        num_workers=num_workers,
        gamma=gamma,
        eta=eta,
        action_type=action_type,
    )

    batch_size = 5
    mock_input = [
        torch.rand((batch_size * num_workers, *D_in)),
        torch.rand((batch_size * num_workers, 1)),
        torch.rand((batch_size * num_workers, *D_in)),
    ]
    out = net(*mock_input, update_ri=True)

    assert out[0].shape == (batch_size * num_workers,)
    assert out[1].shape == ()
    assert out[2].shape == ()

    mock_input = [
        torch.rand((batch_size, *D_in)),
        torch.rand((batch_size, 1)),
        torch.rand((batch_size, *D_in)),
    ]
    out = net(*mock_input, update_ri=False)

    assert out[0].shape == (batch_size,)
    assert out[1].shape == ()
    assert out[2].shape == ()


def test_continuous_icm_cnn_call():
    D_in, D_out, D_hidden = [3, 36, 36], 3, 4
    num_workers, gamma, eta = 2, 0.99, 0.1
    action_type = "continuous"
    net = ICM_CNN(
        D_in=D_in,
        D_out=D_out,
        D_hidden=D_hidden,
        num_workers=num_workers,
        gamma=gamma,
        eta=eta,
        action_type=action_type,
    )

    batch_size = 5
    mock_input = [
        torch.rand((batch_size * num_workers, *D_in)),
        torch.rand((batch_size * num_workers, D_out)),
        torch.rand((batch_size * num_workers, *D_in)),
    ]
    out = net(*mock_input, update_ri=True)

    assert out[0].shape == (batch_size * num_workers,)
    assert out[1].shape == ()
    assert out[2].shape == ()

    mock_input = [
        torch.rand((batch_size, *D_in)),
        torch.rand((batch_size, D_out)),
        torch.rand((batch_size, *D_in)),
    ]
    out = net(*mock_input, update_ri=False)

    assert out[0].shape == (batch_size,)
    assert out[1].shape == ()
    assert out[2].shape == ()


def test_discrete_icm_multi_call():
    D_in, D_out, D_hidden = [[3, 36, 36], 2], 3, 4
    num_workers, gamma, eta = 2, 0.99, 0.1
    action_type = "discrete"
    net = ICM_Multi(
        D_in=D_in,
        D_out=D_out,
        D_hidden=D_hidden,
        num_workers=num_workers,
        gamma=gamma,
        eta=eta,
        action_type=action_type,
    )

    batch_size = 5
    mock_input = [
        [
            torch.rand((batch_size * num_workers, *D_in[0])),
            torch.rand((batch_size * num_workers, D_in[1])),
        ],
        torch.rand((batch_size * num_workers, 1)),
        [
            torch.rand((batch_size * num_workers, *D_in[0])),
            torch.rand((batch_size * num_workers, D_in[1])),
        ],
    ]
    out = net(*mock_input, update_ri=True)

    assert out[0].shape == (batch_size * num_workers,)
    assert out[1].shape == ()
    assert out[2].shape == ()

    mock_input = [
        [torch.rand((batch_size, *D_in[0])), torch.rand((batch_size, D_in[1]))],
        torch.rand((batch_size, 1)),
        [torch.rand((batch_size, *D_in[0])), torch.rand((batch_size, D_in[1]))],
    ]
    out = net(*mock_input, update_ri=False)

    assert out[0].shape == (batch_size,)
    assert out[1].shape == ()
    assert out[2].shape == ()


def test_continuous_icm_multi_call():
    D_in, D_out, D_hidden = [[3, 36, 36], 2], 3, 4
    num_workers, gamma, eta = 2, 0.99, 0.1
    action_type = "continuous"
    net = ICM_Multi(
        D_in=D_in,
        D_out=D_out,
        D_hidden=D_hidden,
        num_workers=num_workers,
        gamma=gamma,
        eta=eta,
        action_type=action_type,
    )

    batch_size = 5
    mock_input = [
        [
            torch.rand((batch_size * num_workers, *D_in[0])),
            torch.rand((batch_size * num_workers, D_in[1])),
        ],
        torch.rand((batch_size * num_workers, D_out)),
        [
            torch.rand((batch_size * num_workers, *D_in[0])),
            torch.rand((batch_size * num_workers, D_in[1])),
        ],
    ]
    out = net(*mock_input, update_ri=True)

    assert out[0].shape == (batch_size * num_workers,)
    assert out[1].shape == ()
    assert out[2].shape == ()

    mock_input = [
        [torch.rand((batch_size, *D_in[0])), torch.rand((batch_size, D_in[1]))],
        torch.rand((batch_size, D_out)),
        [torch.rand((batch_size, *D_in[0])), torch.rand((batch_size, D_in[1]))],
    ]
    out = net(*mock_input, update_ri=False)

    assert out[0].shape == (batch_size,)
    assert out[1].shape == ()
    assert out[2].shape == ()
