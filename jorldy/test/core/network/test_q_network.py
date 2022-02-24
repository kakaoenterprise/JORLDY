import torch

from core.network.q_network import DiscreteQ_Network, ContinuousQ_Network


def test_discrete_q_network_call():
    D_in, D_out, D_hidden = 2, 3, 4
    net = DiscreteQ_Network(D_in=D_in, D_out=D_out, D_hidden=D_hidden)

    batch_size = 5
    mock_input = torch.rand((batch_size, D_in))
    out = net(mock_input)

    assert out.shape == (batch_size, D_out)


def test_continuous_q_network_call():
    D_in1, D_in2, D_hidden = 2, 3, 4
    net = ContinuousQ_Network(D_in1=D_in1, D_in2=D_in2, D_hidden=D_hidden)

    batch_size = 5
    mock_input1 = torch.rand((batch_size, D_in1))
    mock_input2 = torch.rand((batch_size, D_in2))
    out = net(mock_input1, mock_input2)

    assert out.shape == (batch_size, 1)
