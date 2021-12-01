import torch

from core.network.iqn import IQN


def test_iqn_call():
    D_in, D_out, D_hidden = 2, 3, 4
    D_em, N_sample = 5, 6
    net = IQN(D_in=D_in, D_out=D_out, D_em=D_em, N_sample=N_sample, D_hidden=D_hidden)

    batch_size = 7
    tau_min, tau_max = 0.2, 0.8
    mock_input = torch.rand((batch_size, D_in))
    out = net(mock_input, tau_min=tau_min, tau_max=tau_max)

    assert out[0].shape == (batch_size, N_sample, D_out)
    assert out[1].shape == (batch_size, N_sample, 1)

    assert (tau_min <= out[1]).all()
    assert (tau_max >= out[1]).all()
