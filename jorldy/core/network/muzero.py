import torch
import torch.nn.functional as F

from .base import BaseNetwork
from .utils import orthogonal_init
from .head import Residualblock


class Muzero_Fullyconnected(BaseNetwork):
    """mlp network"""
    def __init__(self, D_in, D_out, D_hidden=512, head="mlp"):
        super(Muzero_Fullyconnected, self).__init__(D_in, D_hidden, head)
        self.num_support = 601
        self.channel_action = 1
        self.D_out = D_out

        # representation -> make hidden state
        self.representation_layer = torch.nn.Linear(D_hidden, D_in)

        # prediction -> make discrete policy and discrete value
        self.policy_layer = torch.nn.Linear(D_in, D_out*self.num_support)
        self.value_distribution_layer = torch.nn.Linear(D_in, self.num_support)

        orthogonal_init(self.policy_layer, "policy")
        orthogonal_init(self.value_distribution_layer, "linear")

        # dynamics -> make reward and next hidden state
        self.reward_distribution_layer = torch.nn.Linear(D_in+self.channel_action, self.num_support)

        orthogonal_init(self.reward_distribution_layer, "linear")

    def representation(self, hidden_state):
        hidden_state = super(Muzero_Fullyconnected, self).forward(hidden_state)
        hidden_state = self.representation_layer(hidden_state)
        hidden_state = F.normalize(hidden_state, dim=0)
        return hidden_state

    def prediction(self, hidden_state):
        # pi(action_distribution)
        policy = self.policy_layer(hidden_state).reshape(self.D_out, self.num_support)
        policy = F.softmax(policy, dim=0)  # .sum(dim=1)

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
    def __init__(self, D_in, D_out, D_hidden=512, head="downsample"):
        super(Muzero_Resnet, self).__init__(D_in, D_hidden, head)
        self.num_residual_block = 16
        self.kernel_size = 1
        self.channel = 256
        self.reduced_channel = 256
        self.channel_action = 1
        self.downsample_img_size = 6
        self.num_support = 601
        self.action_reshape = [1, 1, 6, 6]
        self.D_out = D_out

        # representation -> make hidden state
        self.representation_residual_block = torch.nn.ModuleList(
            [Residualblock(self.channel) for _ in range(self.num_residual_block)]
        )

        # prediction -> make discrete policy and discrete value
        self.prediction_residual_block = torch.nn.ModuleList(
            [Residualblock(self.channel) for _ in range(self.num_residual_block)]
        )
        self.prediction_conv = torch.nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.reduced_channel,
            kernel_size=(self.kernel_size, self.kernel_size)
        )
        self.prediction_policy_layer = torch.nn.Linear(
            in_features=self.channel*(self.downsample_img_size*self.downsample_img_size),
            out_features=D_out*self.num_support
        )
        self.prediction_value_distribution_layer = torch.nn.Linear(
            in_features=self.channel*(self.downsample_img_size*self.downsample_img_size),
            out_features=self.num_support
        )

        orthogonal_init(self.prediction_conv, "conv2d")
        orthogonal_init(self.prediction_policy_layer, "linear")
        orthogonal_init(self.prediction_value_distribution_layer, "linear")

        # dynamics -> make reward and next hidden state
        self.dynamics_conv = torch.nn.Conv2d(
            in_channels=self.channel+self.channel_action,
            out_channels=self.channel,
            kernel_size=(self.kernel_size, self.kernel_size)
        )
        self.dynamics_reward_conv = torch.nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.reduced_channel,
            kernel_size=(self.kernel_size, self.kernel_size)
        )
        self.dynamics_residual_block = torch.nn.ModuleList(
            [Residualblock(self.channel) for _ in range(self.num_residual_block)]
        )
        self.dynamics_reward_distribution_layer = torch.nn.Linear(
            in_features=self.channel*(self.downsample_img_size*self.downsample_img_size),
            out_features=self.num_support
        )

        orthogonal_init(self.dynamics_conv, "conv2d")
        orthogonal_init(self.dynamics_reward_conv, "conv2d")
        orthogonal_init(self.dynamics_reward_distribution_layer, "linear")

    def representation(self, observations, action):
        # down-sampling
        hidden_state = super(Muzero_Resnet, self).forward(observations, action)

        # residual-block
        for block in self.representation_residual_block:
            hidden_state = block(hidden_state)

        # hidden_state_normalized
        hidden_state = F.normalize(hidden_state, dim=0)
        return hidden_state

    def prediction(self, hidden_state):
        # residual-block -> conv -> flatten
        for block in self.prediction_residual_block:
            hidden_state = block(hidden_state)
        hidden_state = self.prediction_conv(hidden_state).flatten()

        # pi(action_distribution)
        policy = self.prediction_policy_layer(hidden_state).reshape(self.D_out, self.num_support)
        policy = F.softmax(policy, dim=0)  # .sum(dim=1)

        # value(distribution)
        value_distribution = self.prediction_value_distribution_layer(hidden_state)
        value_distribution = F.softmax(value_distribution, dim=0)
        return policy, value_distribution

    def dynamics(self, hidden_state, action):
        # hidden_state + action -> conv -> residual-block
        action = torch.reshape(torch.ones_like(hidden_state[0][0]) * action, self.action_reshape)
        combination = torch.cat([hidden_state, action], dim=1)
        combination = self.dynamics_conv(combination)
        for block in self.dynamics_residual_block:
            combination = block(combination)

        # next_hidden_state_normalized
        next_hidden_state = F.normalize(combination, dim=0)

        # conv -> flatten -> reward(distribution)
        combination = self.dynamics_reward_conv(combination).flatten()
        reward_distribution = self.dynamics_reward_distribution_layer(combination)
        reward_distribution = F.softmax(reward_distribution, dim=0)
        return next_hidden_state, reward_distribution


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
    scalar = torch.sign(scalar) * (((torch.sqrt(1 + 4 * 0.001 * (torch.abs(scalar) + 1 + 0.001)) - 1) / (2 * 0.001)) ** 2 - 1)
    return scalar


def scalar2vector(scalar, support_range):
    """initiate target distribution from scalar(batch-2D) & project to learn batch-data"""
    # reduce scale
    scalar = torch.sign(scalar) * (torch.sqrt(torch.abs(scalar) + 1) - 1) + 0.001 * scalar
    scalar = scalar.view(scalar.shape)
    scalar = torch.clamp(scalar, -support_range, support_range)

    # target distribution projection(distribute probability for lower support)
    floor = scalar.floor()
    probability = scalar-floor
    distribution = torch.zeros(scalar.shape[0], scalar.shape[1], 2 * support_range+1).to(scalar.device)
    distribution.scatter_(2, (floor + support_range).long().unsqueeze(-1), (1-probability).unsqueeze(-1))

    # target distribution projection(distribute probability for higher support)
    indexes = floor + support_range + 1
    probability = probability.masked_fill_(2 * support_range < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_range < indexes, 0.0)
    distribution.scatter_(2, indexes.long().unsqueeze(-1), probability.unsqueeze(-1))

    return distribution
