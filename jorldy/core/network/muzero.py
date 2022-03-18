import torch
import torch.nn.functional as F

from .base import BaseNetwork
from .utils import orthogonal_init
from .head import Residualblock


class Muzero_Fullyconnected(BaseNetwork):
    """mlp network"""

    def __init__(self, D_in, D_out, D_hidden=512, head="mlp"):
        super(Muzero_Fullyconnected, self).__init__(D_in, D_hidden, head)
        self.D_out = D_out

        # representation -> make hidden state
        self.representation_layer = torch.nn.Linear(D_hidden, D_in)

        # prediction -> make discrete policy and discrete value
        self.policy_layer = torch.nn.Linear(D_in, D_out * 601)
        self.value_distribution_layer = torch.nn.Linear(D_in, 601)

        orthogonal_init(self.policy_layer, "policy")
        orthogonal_init(self.value_distribution_layer, "linear")

        # dynamics -> make reward and next hidden state
        self.reward_distribution_layer = torch.nn.Linear(D_in + 1, 601)

        orthogonal_init(self.reward_distribution_layer, "linear")

    def representation(self, state):
        # hidden_state
        hidden_state = super(Muzero_Fullyconnected, self).forward(state)
        hidden_state = self.representation_layer(hidden_state)
        hidden_state = F.normalize(hidden_state, dim=0)
        return hidden_state

    def prediction(self, hidden_state):
        # pi(action_distribution)
        policy = self.policy_layer(hidden_state).reshape(self.D_out, 601)
        policy = F.softmax(policy, dim=0)

        # value(action_distribution)
        value_distribution = self.value_distribution_layer(hidden_state)
        return policy, value_distribution

    def dynamics(self, hidden_state, action):
        # hidden_state + action
        combination = torch.cat([hidden_state, torch.unsqueeze(action, dim=0)])

        # next_hidden_state_normalized
        next_hidden_state = F.normalize(combination, dim=0)

        # reward(action_distribution)
        reward_distribution = self.reward_distribution_layer(combination)
        return next_hidden_state, reward_distribution


class Muzero_Resnet(BaseNetwork):
    """residual network"""

    def __init__(self, D_in, D_out, num_stack, D_hidden=512, head="residualblock"):
        super(Muzero_Resnet, self).__init__([256, *D_in[1:]], D_hidden, head)
        self.D_out = D_out

        # representation -> make hidden state
        self.representation_downsample_layer = Downsample(D_in, D_out, num_stack)
        self.representation_resnet = torch.nn.ModuleList([self.head for _ in range(16)])

        # prediction -> make discrete policy and discrete value
        self.prediction_resnet = torch.nn.ModuleList([self.head for _ in range(16)])
        self.prediction_conv = torch.nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(1, 1)
        )
        self.prediction_policy_layer = torch.nn.Linear(
            in_features=256 * (6 * 6), out_features=D_out * 601
        )
        self.prediction_value_distribution_layer = torch.nn.Linear(
            in_features=256 * (6 * 6), out_features=601
        )

        orthogonal_init(self.prediction_conv, "conv2d")
        orthogonal_init(self.prediction_policy_layer, "linear")
        orthogonal_init(self.prediction_value_distribution_layer, "linear")

        # dynamics -> make reward and next hidden state
        self.dynamics_conv = torch.nn.Conv2d(
            in_channels=256 + 1, out_channels=256, kernel_size=(1, 1)
        )
        self.dynamics_reward_conv = torch.nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(1, 1)
        )
        self.dynamics_resnet = torch.nn.ModuleList([self.head for _ in range(16)])
        self.dynamics_reward_distribution_layer = torch.nn.Linear(
            in_features=256 * (6 * 6), out_features=601
        )


        self.encode = torch.nn.Conv2d(
            65,
            128,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        orthogonal_init(self.dynamics_conv, "conv2d")
        orthogonal_init(self.dynamics_reward_conv, "conv2d")
        orthogonal_init(self.dynamics_reward_distribution_layer, "linear")

    def representation(self, observations, action):
        # downsample
        hidden_state = self.representation_downsample_layer(observations, action)
        
        # resnet
        for block in self.representation_resnet:
            hidden_state = block(hidden_state)

        # hidden_state_normalized
        hidden_state = F.normalize(hidden_state, dim=0)
        return hidden_state

    def prediction(self, hidden_state):
        # resnet -> conv -> flatten
        for block in self.prediction_resnet:
            hidden_state = block(hidden_state)
        hidden_state = self.prediction_conv(hidden_state).flatten()

        # pi(action_distribution)
        policy = self.prediction_policy_layer(hidden_state).reshape(self.D_out, 601)
        policy = F.softmax(policy, dim=0)

        # value(distribution)
        value_distribution = self.prediction_value_distribution_layer(hidden_state)
        value_distribution = F.softmax(value_distribution, dim=0)
        return policy, value_distribution

    def dynamics(self, hidden_state, action):
        # hidden_state + action -> conv -> resnet
        action = torch.reshape(
            torch.ones_like(hidden_state[0][0]) * action, [1, 1, 6, 6]
        )
        combination = torch.cat([hidden_state, action], dim=1)
        combination = self.dynamics_conv(combination)
        for block in self.dynamics_resnet:
            combination = block(combination)

        # next_hidden_state_normalized
        next_hidden_state = F.normalize(combination, dim=0)

        # conv -> flatten -> reward(distribution)
        combination = self.dynamics_reward_conv(combination).flatten()
        reward_distribution = self.dynamics_reward_distribution_layer(combination)
        reward_distribution = F.softmax(reward_distribution, dim=0)
        return next_hidden_state, reward_distribution


    # codes modified from https://github.com/werner-duvaud/muzero-general
    @staticmethod
    def vector2scalar(probabilities, support_range):
        """prediction value & dynamics reward output(vector:distribution) -> output(scalar:value)"""
        # get supports
        support = (
            torch.tensor([x for x in range(-support_range, support_range + 1)])
            .expand(probabilities.shape)
            .float()
            .to(device=probabilities.device)
        )

        # convert to scalar
        scalar = torch.sum(support * probabilities, dim=0, keepdim=True)

        # Invertible scaling
        eps = 0.001
        scalar = torch.sign(scalar) * (
            (
                (torch.sqrt(1 + 4 * eps * (torch.abs(scalar) + 1 + eps)) - 1)
                / (2 * eps)
            )
            ** 2
            - 1
        )
        return scalar


    # codes modified from https://github.com/werner-duvaud/muzero-general
    @staticmethod
    def scalar2vector(scalar, support_range):
        """initiate target distribution from scalar(batch-2D) & project to learn batch-data"""
        # reduce scale
        scalar = (
            torch.sign(scalar) * (torch.sqrt(torch.abs(scalar) + 1) - 1) + 0.001 * scalar
        )
        scalar = scalar.view(scalar.shape)
        scalar = torch.clamp(scalar, -support_range, support_range)

        # target distribution projection(distribute probability for lower support)
        floor = scalar.floor()
        probability = scalar - floor
        distribution = torch.zeros(
            scalar.shape[0], scalar.shape[1], 2 * support_range + 1
        ).to(scalar.device)
        distribution.scatter_(
            2, (floor + support_range).long().unsqueeze(-1), (1 - probability).unsqueeze(-1)
        )

        # target distribution projection(distribute probability for higher support)
        indexes = floor + support_range + 1
        probability = probability.masked_fill_(2 * support_range < indexes, 0.0)
        indexes = indexes.masked_fill_(2 * support_range < indexes, 0.0)
        distribution.scatter_(2, indexes.long().unsqueeze(-1), probability.unsqueeze(-1))

        return distribution


class Downsample(torch.nn.Module):
    def __init__(self, D_in, D_out, num_stack):
        super(Downsample, self).__init__()
        self.action_divisor = D_out

        D_in[0] = num_stack

        self.conv_1 = torch.nn.Conv2d(
            in_channels=D_in[0],
            out_channels=D_in[0],
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
        self.resnet_1 = torch.nn.ModuleList([Residualblock([128, *D_in[1:]]) for _ in range(2)])
        self.resnet_2 = torch.nn.ModuleList(
            [Residualblock([256, *D_in[1:]]) for _ in range(3)]
        )
        self.resnet_3 = torch.nn.ModuleList(
            [Residualblock([256, *D_in[1:]]) for _ in range(3)]
        )

        self.encode = torch.nn.Conv2d(
            65,
            128,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

    def forward(self, observations, action):
        # observation, action : input -> normalize -> concatenate
        #observations = torch.reshape(observations, [1, 96, 96, 96])
        observations = F.normalize(observations)
        #action = torch.reshape(action, [1, 32, 96, 96])
        action /= self.action_divisor
        x = torch.cat([observations, action], dim=1)

        # down-sampling : conv -> resnet -> pooling
        x = self.conv_1(x)
        x = self.encode(x)
        for block in self.resnet_1:
            x = block(x)

        x = self.conv_2(x)
        for block in self.resnet_2:
            x = block(x)

        x = F.avg_pool2d(x, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        for block in self.resnet_3:
            x = block(x)
        x = F.avg_pool2d(x, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        return x
