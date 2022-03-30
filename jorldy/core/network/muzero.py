import torch
import torch.nn.functional as F

from .base import BaseNetwork
from .utils import orthogonal_init, Converter
from .head import Residualblock


class Muzero_mlp(BaseNetwork):
    """mlp network"""

    def __init__(self, D_in, D_out, num_stack, support, D_hidden=256, head="mlp"):
        super(Muzero_mlp, self).__init__(D_in*(num_stack+1)+D_out*num_stack, D_hidden, head)
        self.D_in = D_in
        self.D_out = D_out
        self.D_hidden = D_hidden
        self.converter = Converter(support)

        # representation -> make hidden state
        self.hs_l = torch.nn.Linear(D_hidden, D_hidden)

        # prediction -> make discrete policy and discrete value
        self.pi_l = torch.nn.Linear(D_hidden, D_out)
        self.vd_l = torch.nn.Linear(D_hidden, (support << 1) + 1)

        orthogonal_init(self.pi_l, "policy")
        orthogonal_init(self.vd_l, "linear")

        # dynamics -> make reward and next hidden state
        self.rd_l = torch.nn.Linear(D_hidden+D_out, (support << 1) + 1)
        self.next_hs_l = torch.nn.Linear(D_hidden+D_out, D_hidden)

        orthogonal_init(self.rd_l, "linear")

    def representation(self, obs, a):
        a = F.one_hot(a.long(), num_classes=self.D_out).view([obs.size(0), -1])
        obs_a = torch.cat([obs, a], dim=-1)
        hs = super(Muzero_mlp, self).forward(obs_a)
        hs = self.hs_l(hs)
        return hs

    def prediction(self, hs):
        # pi(action_distribution)
        pi = self.pi_l(hs)
        pi = F.relu(pi)
        pi = F.log_softmax(pi, dim=-1)

        # value(action_distribution)
        vd = self.vd_l(hs)
        vd = F.relu(vd)
        vd = F.log_softmax(vd, dim=-1)
        return pi, vd

    def dynamics(self, hs, a):
        # hidden_state + action
        a = F.one_hot(a.long(), num_classes=self.D_out).view([hs.size(0), -1])
        hs_a = torch.cat([hs, a], dim=-1)

        # reward(action_distribution)
        rd = self.rd_l(hs_a)
        rd = F.relu(rd)
        rd = F.log_softmax(rd, dim=-1)

        # next_hidden_state_normalized
        next_hs = self.next_hs_l(hs_a)
        return next_hs, rd


class Muzero_Resnet(BaseNetwork):
    """residual network"""

    def __init__(self, D_in, D_out, in_channels, support, D_hidden=256, head="residualblock"):
        super(Muzero_Resnet, self).__init__(D_hidden, D_hidden, head)
        self.D_out = D_out
        self.converter = Converter(support)

        # representation -> make hidden state
        self.hs_down = Downsample(in_channels)
        self.hs_res = torch.nn.ModuleList([self.head for _ in range(16)])

        # prediction -> make discrete policy and discrete value
        self.pred_res = torch.nn.ModuleList([self.head for _ in range(16)])
        self.pred_conv = torch.nn.Conv2d(
            in_channels=D_hidden, out_channels=D_hidden, kernel_size=(1, 1)
        )
        self.pred_pi = torch.nn.Linear(
            in_features=D_hidden * (6 * 6), out_features=D_out
        )
        self.pred_vd = torch.nn.Linear(
            in_features=D_hidden * (6 * 6), out_features=(support << 1) + 1
        )

        orthogonal_init(self.pred_conv, "conv2d")
        orthogonal_init(self.pred_pi, "linear")
        orthogonal_init(self.pred_vd, "linear")

        # dynamics -> make reward and next hidden state
        self.dy_conv = torch.nn.Conv2d(
            in_channels=D_hidden + 1, out_channels=D_hidden, kernel_size=(1, 1)
        )
        self.dy_conv_rd = torch.nn.Conv2d(
            in_channels=D_hidden, out_channels=D_hidden, kernel_size=(1, 1)
        )
        self.dy_res = torch.nn.ModuleList([self.head for _ in range(16)])
        self.dy_rd = torch.nn.Linear(
            in_features=D_hidden * (6 * 6), out_features=(support << 1) + 1
        )

        orthogonal_init(self.dy_conv, "conv2d")
        orthogonal_init(self.dy_conv_rd, "conv2d")
        orthogonal_init(self.dy_rd, "linear")

    def representation(self, obs, a):
        # observation, action : input -> normalize -> concatenate
        obs = F.normalize(obs)
        obs /= self.D_out
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
        pi = self.pred_pi(hs)
        pi = torch.exp(F.log_softmax(pi, dim=-1))

        # value(distribution)
        vd = self.pred_vd(hs)
        vd = torch.exp(F.log_softmax(vd, dim=-1))
        return pi, vd

    def dynamics(self, hs, a):
        # hidden_state + action -> conv -> resnet
        a = torch.broadcast_to(a.unsqueeze(dim=-1).unsqueeze(dim=-1), [a.size(0), 1, 6, 6])
        hs_a = torch.cat([hs, a], dim=1)
        hs_a = self.dy_conv(hs_a)
        for block in self.dy_res:
            hs_a = block(hs_a)

        # next_hidden_state_normalized
        next_hs = F.normalize(hs_a, dim=0)

        # conv -> flatten -> reward(distribution)
        hs_a = self.dy_conv_rd(hs_a).reshape(hs.size(0), -1)
        rd = self.dy_rd(hs_a)
        rd = torch.exp(F.log_softmax(rd, dim=-1))
        return next_hs, rd


class Downsample(torch.nn.Module):
    def __init__(self, in_channels):
        super(Downsample, self).__init__()

        self.conv_1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )
        self.conv_2 = torch.nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )

        # resnet
        self.res_1 = torch.nn.ModuleList([Residualblock(128) for _ in range(2)])
        self.res_2 = torch.nn.ModuleList(
            [Residualblock(256) for _ in range(3)]
        )
        self.res_3 = torch.nn.ModuleList(
            [Residualblock(256) for _ in range(3)]
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
