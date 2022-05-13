import torch
import torch.nn.functional as F
import numpy as np

from .base import BaseNetwork
from .utils import orthogonal_init, Converter


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

        self.hs_res = torch.nn.Sequential(*[MLP_Residualblock(D_hidden, D_hidden) for _ in range(num_rb)])

        # prediction -> make discrete policy and discrete value
        self.pred_res = torch.nn.Sequential(*[MLP_Residualblock(D_hidden, D_hidden) for _ in range(num_rb)])

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
        self.dy_res = torch.nn.Sequential(*[MLP_Residualblock(D_hidden, D_hidden) for _ in range(num_rb)])

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
        
        assert D_in[1] >= 16 and D_in[2] >= 16
        
        self.D_in = D_in
        self.D_out = D_out
        self.converter = Converter(support)
        self.state_channel = D_in[0]
        self.action_channel = 1

        D_stack = self.state_channel * (num_stack + 1) + self.action_channel * num_stack

        kernel = (1, 1)
        padding = (0, 0)
        stride = (1, 1)

        # representation -> make hidden state
        self.hs_down = Downsample(D_stack, num_rb, D_hidden)
        self.hs_res = torch.nn.Sequential(*[CONV_Residualblock(D_hidden, D_hidden) for _ in range(num_rb)])

        filter_info = self.hs_down.get_filter_info()
        dim1 = (
            (
                (D_in[-2] - filter_info["kernel"][0] + (filter_info["padding"][0] << 1))
                // filter_info["stride"][0]
                + 1
            ),
            (
                (D_in[-1] - filter_info["kernel"][1] + (filter_info["padding"][1] << 1))
                // filter_info["stride"][1]
                + 1
            ),
        )
        dim2 = (
            (
                (dim1[0] - filter_info["kernel"][0] + (filter_info["padding"][0] << 1))
                // filter_info["stride"][0]
                + 1
            ),
            (
                (dim1[1] - filter_info["kernel"][1] + (filter_info["padding"][1] << 1))
                // filter_info["stride"][1]
                + 1
            ),
        )
        dim3 = (
            dim2[0] // filter_info["stride"][0],
            dim2[1] // filter_info["stride"][0],
        )
        dim4 = (
            dim3[0] // filter_info["stride"][1],
            dim3[1] // filter_info["stride"][1],
        )
        self.down_size = (dim4[0], dim4[1])

        # prediction -> make discrete policy and discrete value
        self.pred_res = torch.nn.Sequential(*[CONV_Residualblock(D_hidden, D_hidden) for _ in range(num_rb)])
        self.pred_conv = torch.nn.Conv2d(
            in_channels=D_hidden,
            out_channels=D_hidden,
            kernel_size=kernel,
            padding=padding,
            stride=stride,
        )
        self.pred_pi_1 = torch.nn.Linear(
            in_features=D_hidden * (self.down_size[0] * self.down_size[1]),
            out_features=D_hidden,
        )
        self.pred_pi_2 = torch.nn.Linear(in_features=D_hidden, out_features=D_out)
        # self.pred_pi_3 = torch.nn.Linear(in_features=D_hidden, out_features=D_out)
        self.pred_vd_1 = torch.nn.Linear(
            in_features=D_hidden * (self.down_size[0] * self.down_size[1]),
            out_features=D_hidden,
        )
        self.pred_vd_2 = torch.nn.Linear(in_features=D_hidden, out_features=(support << 1) + 1)
        # self.pred_vd_3 = torch.nn.Linear(
        #     in_features=D_hidden, out_features=(support << 1) + 1
        # )

        orthogonal_init(self.pred_conv, "conv2d")
        orthogonal_init(self.pred_pi_1, "linear")
        orthogonal_init(self.pred_pi_2, "linear")
        # orthogonal_init(self.pred_pi_3, "linear")
        orthogonal_init(self.pred_vd_1, "linear")
        orthogonal_init(self.pred_vd_2, "linear")
        # orthogonal_init(self.pred_vd_3, "linear")

        # dynamics -> make reward and next hidden state
        self.dy_conv = torch.nn.Conv2d(
            in_channels=D_hidden + 1,
            out_channels=D_hidden,
            kernel_size=kernel,
            padding=padding,
            stride=stride,
        )
        self.dy_conv_rd = torch.nn.Conv2d(
            in_channels=D_hidden,
            out_channels=D_hidden,
            kernel_size=kernel,
            padding=padding,
            stride=stride,
        )
        self.dy_res = torch.nn.Sequential(*[CONV_Residualblock(D_hidden, D_hidden) for _ in range(num_rb)])
        self.dy_rd_1 = torch.nn.Linear(
            in_features=D_hidden * (self.down_size[0] * self.down_size[1]),
            out_features=D_hidden,
        )
        self.dy_rd_2 = torch.nn.Linear(in_features=D_hidden, out_features=(support << 1) + 1)
        # self.dy_rd_3 = torch.nn.Linear(
        #     in_features=D_hidden, out_features=(support << 1) + 1
        # )

        orthogonal_init(self.dy_conv, "conv2d")
        orthogonal_init(self.dy_conv_rd, "conv2d")
        orthogonal_init(self.dy_rd_1, "linear")
        orthogonal_init(self.dy_rd_2, "linear")
        # orthogonal_init(self.dy_rd_3, "linear")

    def representation(self, obs, a):
        # observation, action : normalize -> concatenate -> input
        obs = torch.div(obs, 255.0)
        a = torch.div(a, self.D_out).view([*a.size()[:2], 1, 1])
        a = torch.broadcast_to(a, [*a.size()[:2], *self.D_in[1:]])
        obs_a = torch.cat([obs, a], dim=1)

        # downsample
        hs = self.hs_down(obs_a)

        # resnet
        for block in self.hs_res:
            hs = block(hs)

        # hidden_state_normalize
        original_size = hs.size()
        hs = hs.view([original_size[0], -1])
        hs_max = hs.max(1)[0].view(original_size[0], -1)
        hs_min = hs.min(1)[0].view(original_size[0], -1)
        hs_scale = hs_max - hs_min
        hs_scale[hs_scale < 1e-5] += 1e-5
        hs_norm = (hs - hs_min) / hs_scale
        hs_norm = hs_norm.view(original_size)

        return hs_norm

    def prediction(self, hs):
        # resnet -> conv -> flatten
        for block in self.pred_res:
            hs = block(hs)
        hs = self.pred_conv(hs)
        hs = F.leaky_relu(hs)
        hs = hs.view(hs.size(0), -1)

        # pi(action_distribution)
        pi = self.pred_pi_1(hs)
        pi = F.leaky_relu(pi)
        pi = self.pred_pi_2(pi)
        # pi = F.leaky_relu(pi)
        # pi = self.pred_pi_3(pi)
        pi = F.log_softmax(pi, dim=-1)

        # value(distribution)
        vd = self.pred_vd_1(hs)
        vd = F.leaky_relu(vd)
        vd = self.pred_vd_2(vd)
        # vd = F.leaky_relu(vd)
        # vd = self.pred_vd_3(vd)
        vd = F.log_softmax(vd, dim=-1)

        return pi, vd

    def dynamics(self, hs, a):
        # hidden_state + action -> conv -> resnet
        a = torch.div(a, self.D_out).view([*a.size()[:2], 1, 1])
        a = torch.broadcast_to(a, [*a.size()[:2], *self.down_size])
        hs_a = torch.cat([hs, a], dim=1)
        next_hs = self.dy_conv(hs_a)
        next_hs = F.leaky_relu(next_hs)
        for block in self.dy_res:
            next_hs = block(next_hs)

        # next_hidden_state_normalize
        original_size = next_hs.size()
        next_hs = next_hs.view([original_size[0], -1])
        next_hs_max = next_hs.max(1)[0].view(original_size[0], -1)
        next_hs_min = next_hs.min(1)[0].view(original_size[0], -1)
        next_hs_scale = next_hs_max - next_hs_min
        next_hs_scale[next_hs_scale < 1e-5] += 1e-5
        next_hs_norm = (next_hs - next_hs_min) / next_hs_scale
        next_hs_norm = next_hs_norm.view(original_size)

        # conv(norm) -> flatten -> reward(distribution)
        rd = self.dy_conv_rd(next_hs_norm).view(next_hs_norm.size(0), -1)
        rd = F.leaky_relu(rd)
        rd = self.dy_rd_1(rd)
        rd = F.leaky_relu(rd)
        rd = self.dy_rd_2(rd)
        # rd = F.leaky_relu(rd)
        # rd = self.dy_rd_3(rd)
        rd = F.log_softmax(rd, dim=-1)

        return next_hs_norm, rd


class Downsample(torch.nn.Module):
    def __init__(self, in_channels, num_rb, D_hidden):
        super(Downsample, self).__init__()

        self.kernel = (3, 3)
        self.stride = (2, 2)
        self.padding = (1, 1)

        # conv
        self.conv_1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=(D_hidden >> 1),
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
            bias=False,
        )
        self.conv_2 = torch.nn.Conv2d(
            in_channels=(D_hidden >> 1),
            out_channels=D_hidden,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
            bias=False,
        )

        # resnet
        self.res_1 = torch.nn.Sequential(
            *[CONV_Residualblock(D_hidden >> 1) for _ in range(num_rb)]
        )
        self.res_2 = torch.nn.Sequential(
            *[CONV_Residualblock(D_hidden) for _ in range(num_rb)]
        )
        self.res_3 = torch.nn.Sequential(
            *[CONV_Residualblock(D_hidden) for _ in range(num_rb)]
        )

    def forward(self, obs_a):
        # down-sampling : conv -> resnet -> pooling
        obs_a = self.conv_1(obs_a)
        obs_a = F.leaky_relu(obs_a)
        for block in self.res_1:
            obs_a = block(obs_a)
            
        obs_a = self.conv_2(obs_a)
        obs_a = F.leaky_relu(obs_a)
        for block in self.res_2:
            obs_a = block(obs_a)
            
        obs_a = F.avg_pool2d(
            obs_a, kernel_size=self.kernel, stride=self.stride, padding=self.padding
        )
        for block in self.res_3:
            obs_a = block(obs_a)
            
        obs_a = F.avg_pool2d(
            obs_a, kernel_size=self.kernel, stride=self.stride, padding=self.padding
        )

        return obs_a

    def get_filter_info(self):
        return {
            "kernel": self.kernel,
            "stride": self.stride,
            "padding": self.padding,
        }
    

class MLP_Residualblock(torch.nn.Module):
    def __init__(self, D_in, D_hidden=256):
        super(MLP_Residualblock, self).__init__()

        self.l1 = torch.nn.Linear(D_in, D_hidden)
        self.ln1 = torch.nn.LayerNorm(D_hidden)
        self.l2 = torch.nn.Linear(D_hidden, D_in)
        self.ln2 = torch.nn.LayerNorm(D_hidden)

    def forward(self, x):
        x_res = F.relu(self.ln1(self.l1(x)))
        x_res = self.ln2(self.l2(x))
        x_res += x
        x = F.relu(x_res)
        return x


# muzero atari head
class CONV_Residualblock(torch.nn.Module):
    def __init__(self, D_in, D_hidden=256):
        super(CONV_Residualblock, self).__init__()

        self.c1 = torch.nn.Conv2d(
            in_channels=D_in,
            out_channels=D_in,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.b1 = torch.nn.BatchNorm2d(num_features=D_in)
        self.c2 = torch.nn.Conv2d(
            in_channels=D_in,
            out_channels=D_in,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.b2 = torch.nn.BatchNorm2d(num_features=D_in)

    def forward(self, x):
        x_res = self.c1(x)
        x_res = self.b1(x_res)
        x_res = F.relu(x_res)
        x_res = self.c2(x_res)
        x_res = self.b2(x_res)
        x_res += x
        x = F.relu(x_res)
        return x
        