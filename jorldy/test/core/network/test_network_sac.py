import torch

from core.network.sac import SAC_Critic


def test_sac_critic_call():
    D_in1, D_in2, D_hidden = 2, 3, 4
    net = SAC_Critic(D_in1=D_in1, D_in2=D_in2, D_hidden=D_hidden)

    batch_size = 5
    mock_input1 = torch.rand((batch_size, D_in1))
    mock_input2 = torch.rand((batch_size, D_in2))
    out = net(mock_input1, mock_input2)

    assert out[0].shape == (batch_size, 1)
    assert out[1].shape == (batch_size, 1)
