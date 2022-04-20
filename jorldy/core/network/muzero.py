import torch
import torch.nn.functional as F

from .base import BaseNetwork
from .utils import orthogonal_init, Converter
from .head import Residualblock


class Muzero_mlp(BaseNetwork):
    """mlp network"""

    def __init__(
        self,
        D_in,
        D_out,
        num_stack,
        support,
        num_rb=10,
        D_hidden=256,
        head="mlp_residualblock",
    ):
        super(Muzero_mlp, self).__init__(D_hidden, D_hidden, head)
        self.D_in = D_in
        self.D_out = D_out
        self.D_hidden = D_hidden
        self.converter = Converter(support)

        D_stack = D_in * (num_stack + 1) + num_stack

        # representation -> make hidden state
        self.hs_l1 = torch.nn.Linear(D_stack, D_hidden)
        self.hs_ln1 = torch.nn.LayerNorm(D_hidden)

        self.hs_res = torch.nn.Sequential(*[self.head for _ in range(num_rb)])

        # prediction -> make discrete policy and discrete value
        self.pred_res = torch.nn.Sequential(*[self.head for _ in range(num_rb)])

        self.pi_l1 = torch.nn.Linear(D_hidden, D_hidden)
        self.pi_l2 = torch.nn.Linear(D_hidden, D_hidden)
        self.pi_l3 = torch.nn.Linear(D_hidden, D_out)
        self.vd_l1 = torch.nn.Linear(D_hidden, D_hidden)
        self.vd_l2 = torch.nn.Linear(D_hidden, D_hidden)
        self.vd_l3 = torch.nn.Linear(D_hidden, (support << 1) + 1)

        orthogonal_init(self.pi_l1, "policy")
        orthogonal_init(self.pi_l2, "policy")
        orthogonal_init(self.pi_l3, "policy")
        orthogonal_init(self.vd_l1, "linear")
        orthogonal_init(self.vd_l2, "linear")
        orthogonal_init(self.vd_l3, "linear")

        # dynamics -> make reward and next hidden state
        self.dy_l1 = torch.nn.Linear(D_hidden + D_out, D_hidden)
        self.dy_res = torch.nn.Sequential(*[self.head for _ in range(num_rb)])

        self.rd_l1 = torch.nn.Linear(D_hidden, D_hidden)
        self.rd_l2 = torch.nn.Linear(D_hidden, (support << 1) + 1)

        orthogonal_init(self.dy_l1, "linear")
        orthogonal_init(self.rd_l1, "linear")
        orthogonal_init(self.rd_l2, "linear")

    def representation(self, obs, a):
        obs_a = torch.cat([obs, a], dim=-1)

        hs = self.hs_l1(obs_a)
        hs = self.hs_ln1(hs)
        hs = self.hs_res(hs)

        hs = F.normalize(hs)
        return hs

    def prediction(self, hs):
        hs = self.pred_res(hs)

        # pi(action_distribution)
        pi = self.pi_l1(hs)
        pi = F.leaky_relu(pi)
        pi = self.pi_l2(pi)
        pi = F.leaky_relu(pi)
        pi = self.pi_l3(pi)
        pi = F.log_softmax(pi, dim=-1)

        # value(action_distribution)
        vd = self.vd_l1(hs)
        vd = F.leaky_relu(vd)
        vd = self.vd_l2(vd)
        vd = F.leaky_relu(vd)
        vd = self.vd_l3(vd)
        vd = F.log_softmax(vd, dim=-1)
        return pi, vd

    def dynamics(self, hs, a):
        # hidden_state + action
        a = F.one_hot(a.long(), num_classes=self.D_out).view([hs.size(0), -1])
        hs_a = torch.cat([hs, a], dim=-1)

        hs_a = F.relu(self.dy_l1(hs_a))
        hs_a = self.dy_res(hs_a)

        # next_hidden_state_normalized
        next_hs = F.normalize(hs_a)
        
        # reward(action_distribution)
        rd = self.rd_l1(hs_a)
        rd = F.leaky_relu(rd)
        rd = self.rd_l2(rd)
        rd = F.log_softmax(rd, dim=-1)
        
        return next_hs, rd


class Muzero_Resnet(BaseNetwork):
    """residual network"""

    def __init__(
        self,
        D_in,
        D_out,
        num_stack,
        support,
        num_rb=16,
        D_hidden=256,
        head="residualblock",
    ):
        super(Muzero_Resnet, self).__init__(D_hidden, D_hidden, head)
        self.D_out = D_out
        self.converter = Converter(support)
        self.state_channel = D_in[0]
        self.action_channel = 1

        D_stack = self.state_channel * (num_stack + 1) + self.action_channel * num_stack
        kernel_size = (1, 1)
        padding = (0, 0)
        stride = (1, 1)

        assert D_in[1] == D_in[2], "Image width must have same size with height"

        # representation -> make hidden state
        self.hs_down = Downsample(D_stack, num_rb, D_hidden)
        self.hs_res = torch.nn.ModuleList([self.head for _ in range(num_rb)])

        # prediction -> make discrete policy and discrete value
        self.pred_res = torch.nn.ModuleList([self.head for _ in range(num_rb)])
        self.pred_conv = torch.nn.Conv2d(
            in_channels=D_hidden,
            out_channels=D_hidden,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.pred_pi_1 = torch.nn.Linear(
            in_features=(
                (D_in[1] - kernel_size[0] + (padding[0] << 1)) // stride[0] + 1
            )
            ** 2,
            out_features=D_hidden,
        )
        self.pred_pi_2 = torch.nn.Linear(in_features=D_hidden, out_features=D_hidden)
        self.pred_pi_3 = torch.nn.Linear(in_features=D_hidden, out_features=D_out)
        self.pred_vd_1 = torch.nn.Linear(
            in_features=(
                (D_in[1] - kernel_size[0] + (padding[0] << 1)) // stride[0] + 1
            )
            ** 2,
            out_features=D_hidden,
        )
        self.pred_vd_2 = torch.nn.Linear(in_features=D_hidden, out_features=D_hidden)
        self.pred_vd_3 = torch.nn.Linear(
            in_features=D_hidden, out_features=(support << 1) + 1
        )

        orthogonal_init(self.pred_conv, "conv2d")
        orthogonal_init(self.pred_pi_1, "linear")
        orthogonal_init(self.pred_pi_2, "linear")
        orthogonal_init(self.pred_pi_3, "linear")
        orthogonal_init(self.pred_vd_1, "linear")
        orthogonal_init(self.pred_vd_2, "linear")
        orthogonal_init(self.pred_vd_3, "linear")

        # dynamics -> make reward and next hidden state
        self.dy_conv = torch.nn.Conv2d(
            in_channels=D_hidden + 1,
            out_channels=D_hidden,
            kernel_size=(1, 1),
            padding=padding,
            stride=stride,
        )
        self.dy_conv_rd = torch.nn.Conv2d(
            in_channels=D_hidden,
            out_channels=D_hidden,
            kernel_size=(1, 1),
            padding=padding,
            stride=stride,
        )
        self.dy_res = torch.nn.ModuleList([self.head for _ in range(num_rb)])
        self.dy_rd_1 = torch.nn.Linear(
            in_features=(
                (D_in[1] - kernel_size[0] + (padding[0] << 1)) // stride[0] + 1
            )
            ** 2,
            out_features=D_hidden,
        )
        self.dy_rd_2 = torch.nn.Linear(in_features=D_hidden, out_features=D_hidden)
        self.dy_rd_3 = torch.nn.Linear(
            in_features=D_hidden, out_features=(support << 1) + 1
        )

        orthogonal_init(self.dy_conv, "conv2d")
        orthogonal_init(self.dy_conv_rd, "conv2d")
        orthogonal_init(self.dy_rd_1, "linear")
        orthogonal_init(self.dy_rd_2, "linear")
        orthogonal_init(self.dy_rd_3, "linear")

    def representation(self, obs, a):
        # observation, action : input -> normalize -> concatenate
        obs /= 255.0
        a /= self.D_out
        obs_a = torch.cat([obs, a], dim=1)

        # downsample
        hs = self.hs_down(obs_a)

        # resnet
        for block in self.hs_res:
            hs = block(hs)

        # hidden_state_normalized
        hs = F.normalize(hs, dim=0)
        return hs

    def prediction(self, hs):
        # resnet -> conv -> flatten
        for block in self.pred_res:
            hs = block(hs)
        hs = self.pred_conv(hs)
        hs = hs.reshape(hs.size(0), -1)

        # pi(action_distribution)
        pi = self.pred_pi_1(hs)
        pi = F.leaky_relu(pi)
        pi = self.pred_pi_2(pi)
        pi = F.leaky_relu(pi)
        pi = self.pred_pi_3(pi)
        pi = F.log_softmax(pi, dim=-1)

        # value(distribution)
        vd = self.pred_vd_1(hs)
        vd = F.leaky_relu(vd)
        vd = self.pred_vd_2(vd)
        vd = F.leaky_relu(vd)
        vd = self.pred_vd_3(vd)
        vd = F.log_softmax(vd, dim=-1)
        return pi, vd

    def dynamics(self, hs, a):
        # hidden_state + action -> conv -> resnet
        a = torch.broadcast_to(a.unsqueeze(-1).unsqueeze(-1), [a.size(0), 1, 6, 6])
        hs_a = torch.cat([hs, a], dim=1)
        hs_a = self.dy_conv(hs_a)
        for block in self.dy_res:
            hs_a = block(hs_a)

        # next_hidden_state_normalized
        next_hs = F.normalize(hs_a, dim=0)

        # conv -> flatten -> reward(distribution)
        hs_a = self.dy_conv_rd(hs_a).reshape(hs.size(0), -1)
        rd = self.dy_rd_1(hs_a)
        rd = F.leaky_relu(rd)
        rd = self.dy_rd_2(rd)
        rd = F.leaky_relu(rd)
        rd = self.dy_rd_3(rd)
        rd = F.log_softmax(rd, dim=-1)
        return next_hs, rd


class Downsample(torch.nn.Module):
    def __init__(self, in_channels, num_rb, D_hidden):
        super(Downsample, self).__init__()

        kernel_size = (3, 3)
        stride = (2, 2)
        padding = (1, 1)

        self.conv_1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=(D_hidden >> 1),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.conv_2 = torch.nn.Conv2d(
            in_channels=(D_hidden >> 1),
            out_channels=D_hidden,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        # resnet
        self.res_1 = torch.nn.ModuleList(
            [Residualblock(D_hidden >> 1) for _ in range(num_rb)]
        )
        self.res_2 = torch.nn.ModuleList(
            [Residualblock(D_hidden) for _ in range(num_rb)]
        )
        self.res_3 = torch.nn.ModuleList(
            [Residualblock(D_hidden) for _ in range(num_rb)]
        )

    def forward(self, obs_a):
        # down-sampling : conv -> resnet -> pooling
        obs_a = self.conv_1(obs_a)
        for block in self.res_1:
            obs_a = block(obs_a)
        obs_a = self.conv_2(obs_a)
        for block in self.res_2:
            obs_a = block(obs_a)
        obs_a = F.avg_pool2d(obs_a, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        for block in self.res_3:
            obs_a = block(obs_a)
        obs_a = F.avg_pool2d(obs_a, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        return obs_a
