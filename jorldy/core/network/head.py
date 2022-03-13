import torch
import torch.nn.functional as F
from .utils import orthogonal_init


class MLP(torch.nn.Module):
    def __init__(self, D_in, D_hidden=512):
        super(MLP, self).__init__()

        self.l = torch.nn.Linear(D_in, D_hidden)
        self.D_head_out = D_hidden

        for layer in self.__dict__["_modules"].values():
            orthogonal_init(layer)

    def forward(self, x):
        x = F.relu(self.l(x))
        return x


class CNN(torch.nn.Module):
    def __init__(self, D_in, D_hidden=512):
        super(CNN, self).__init__()

        assert D_in[1] >= 36 and D_in[2] >= 36

        self.conv1 = torch.nn.Conv2d(
            in_channels=D_in[0], out_channels=32, kernel_size=8, stride=4
        )
        dim1 = ((D_in[1] - 8) // 4 + 1, (D_in[2] - 8) // 4 + 1)
        self.conv2 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2
        )
        dim2 = ((dim1[0] - 4) // 2 + 1, (dim1[1] - 4) // 2 + 1)
        self.conv3 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1
        )
        dim3 = ((dim2[0] - 3) // 1 + 1, (dim2[1] - 3) // 1 + 1)

        self.D_head_out = 64 * dim3[0] * dim3[1]

        for layer in self.__dict__["_modules"].values():
            orthogonal_init(layer)

    def forward(self, x):
        x = x / 255.0

        if len(x.shape) == 5:  # sequence
            batch_len, seq_len = x.size(0), x.size(1)

            x = x.reshape(-1, *x.shape[2:])
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(batch_len, seq_len, -1)
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
        return x


class Multi(torch.nn.Module):
    def __init__(self, D_in, D_hidden=512):
        super(Multi, self).__init__()

        D_in_img = D_in[0]
        D_in_vec = D_in[1]

        assert D_in_img[1] >= 36 and D_in_img[2] >= 36

        self.conv1 = torch.nn.Conv2d(
            in_channels=D_in_img[0], out_channels=32, kernel_size=8, stride=4
        )
        dim1 = ((D_in_img[1] - 8) // 4 + 1, (D_in_img[2] - 8) // 4 + 1)
        self.conv2 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2
        )
        dim2 = ((dim1[0] - 4) // 2 + 1, (dim1[1] - 4) // 2 + 1)
        self.conv3 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1
        )
        dim3 = ((dim2[0] - 3) // 1 + 1, (dim2[1] - 3) // 1 + 1)

        self.D_conv_out = 64 * dim3[0] * dim3[1]

        self.l1 = torch.nn.Linear(D_in_vec, D_hidden)
        self.l2 = torch.nn.Linear(D_hidden, D_hidden)

        self.D_mlp_out = D_hidden

        self.D_head_out = self.D_conv_out + self.D_mlp_out

        for layer in self.__dict__["_modules"].values():
            orthogonal_init(layer)

    def forward(self, x):
        x_img = x[0] / 255.0
        x_vec = x[1]

        if len(x_img.shape) == 5:  # sequence
            batch_len, seq_len = x_img.size(0), x_img.size(1)

            x_img = x_img.reshape(-1, *x_img.shape[2:])
            x_img = F.relu(self.conv1(x_img))
            x_img = F.relu(self.conv2(x_img))
            x_img = F.relu(self.conv3(x_img))
            x_img = x_img.reshape(batch_len, seq_len, -1)
        else:
            x_img = F.relu(self.conv1(x_img))
            x_img = F.relu(self.conv2(x_img))
            x_img = F.relu(self.conv3(x_img))
            x_img = x_img.reshape(x_img.size(0), -1)

        x_vec = F.relu(self.l1(x_vec))
        x_vec = F.relu(self.l2(x_vec))

        x_multi = torch.cat((x_img, x_vec), -1)

        return x_multi


class MLP_LSTM(torch.nn.Module):
    def __init__(self, D_in, D_hidden=512):
        super(MLP_LSTM, self).__init__()

        self.l = torch.nn.Linear(D_in, D_hidden)
        self.lstm = torch.nn.LSTM(
            input_size=D_hidden, hidden_size=D_hidden, batch_first=True
        )
        self.D_head_out = D_hidden

    def forward(self, x, hidden_in=None):
        if hidden_in is None:
            hidden_in = (
                torch.zeros(1, x.size(0), self.D_head_out).to(x.device),
                torch.zeros(1, x.size(0), self.D_head_out).to(x.device),
            )

        x = F.relu(self.l(x))
        x, hidden_out = self.lstm(x, hidden_in)

        return x, hidden_in, hidden_out


class CNN_LSTM(torch.nn.Module):
    def __init__(self, D_in, D_hidden=512):
        super(CNN_LSTM, self).__init__()

        assert D_in[1] >= 36 and D_in[2] >= 36

        self.conv1 = torch.nn.Conv2d(
            in_channels=D_in[0], out_channels=32, kernel_size=8, stride=4
        )
        dim1 = ((D_in[1] - 8) // 4 + 1, (D_in[2] - 8) // 4 + 1)
        self.conv2 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2
        )
        dim2 = ((dim1[0] - 4) // 2 + 1, (dim1[1] - 4) // 2 + 1)
        self.conv3 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1
        )
        dim3 = ((dim2[0] - 3) // 1 + 1, (dim2[1] - 3) // 1 + 1)

        self.D_conv_out = 64 * dim3[0] * dim3[1]

        self.lstm = torch.nn.LSTM(
            input_size=self.D_conv_out, hidden_size=D_hidden, batch_first=True
        )

        self.D_head_out = D_hidden

    def forward(self, x, hidden_in=None):
        x = x / 255.0

        seq_len = x.size(1)

        if hidden_in is None:
            hidden_in = (
                torch.zeros(1, x.size(0), self.D_head_out).to(x.device),
                torch.zeros(1, x.size(0), self.D_head_out).to(x.device),
            )

        x = x.reshape(-1, *x.shape[2:])
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, seq_len, self.D_conv_out)
        x, hidden_out = self.lstm(x, hidden_in)

        return x, hidden_in, hidden_out
    

############################################################################################################
# muzero atari head
class Downsample(torch.nn.Module):
    def __init__(self, D_in, D_hidden=512):
        super(Downsample, self).__init__()
        self.D_head_out = [1, 256, 6, 6]  # never use
        self.in_channel = D_in[0]  # D_in = [128, 96, 96] -> in_channel = 128
        self.mid_channel = 128
        self.out_channel = 256
        self.kernel = (3, 3)
        self.stride = (2, 2)
        self.padding = (1, 1)
        self.num_residual_block = [2, 3, 3]
        self.image_reshape = [1, 96, 96, 96]
        self.action_reshape = [1, 32, 96, 96]
        self.action_divisor = 18

        self.conv_1 = torch.nn.Conv2d(
            in_channels=self.in_channel,
            out_channels=self.mid_channel,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
            bias=False
        )
        self.conv_2 = torch.nn.Conv2d(
            in_channels=self.mid_channel,
            out_channels=self.out_channel,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
            bias=False
        )

        # residual block
        self.residual_block_1 = torch.nn.ModuleList(
            [Residualblock(self.in_channel) for _ in range(self.num_residual_block[0])]
        )
        self.residual_block_2 = torch.nn.ModuleList(
            [Residualblock(self.out_channel) for _ in range(self.num_residual_block[1])]
        )
        self.residual_block_3 = torch.nn.ModuleList(
            [Residualblock(self.out_channel) for _ in range(self.num_residual_block[2])]
        )

    def forward(self, observations, action):
        # observation, action : input -> normalize -> concatenate
        observations = torch.reshape(observations, self.image_reshape)
        observations = F.normalize(observations)
        action = torch.reshape(action, self.action_reshape)
        action /= self.action_divisor
        x = torch.cat([observations, action], dim=1)

        # down-sampling : ConvNet -> residual-block -> pooling
        x = self.conv_1(x)
        for block in self.residual_block_1:
            x = block(x)

        x = self.conv_2(x)
        for block in self.residual_block_2:
            x = block(x)

        x = F.avg_pool2d(x, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        for block in self.residual_block_3:
            x = block(x)
        x = F.avg_pool2d(x, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        return x


class Residualblock(torch.nn.Module):
    def __init__(self, channel):
        super(Residualblock, self).__init__()
        self.channel = channel
        self.kernel_size = (3, 3)
        self.stride = (1, 1)
        self.padding = (1, 1)

        self.c1 = torch.nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=False
        )
        self.b1 = torch.nn.BatchNorm2d(num_features=self.channel)
        self.c2 = torch.nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=False
        )
        self.b2 = torch.nn.BatchNorm2d(num_features=self.channel)

    def forward(self, x):
        x_residual_block = self.c1(x)
        x_residual_block = self.b1(x_residual_block)
        x_residual_block = F.relu(x_residual_block)
        x_residual_block = self.c2(x_residual_block)
        x_residual_block = self.b2(x_residual_block)
        x_residual_block += x
        x = F.relu(x_residual_block)
        return x
############################################################################################################


import os, sys, inspect, re
from collections import OrderedDict

working_path = os.path.dirname(os.path.realpath(__file__))
head_dict = {}
naming_rule = lambda x: re.sub("([a-z])([A-Z])", r"\1_\2", x).lower()
for class_name, _class in inspect.getmembers(sys.modules[__name__], inspect.isclass):
    if __name__ in str(_class):
        head_dict[naming_rule(class_name)] = _class

head_dict = OrderedDict(sorted(head_dict.items()))
with open(os.path.join(working_path, "_head_dict.txt"), "w") as f:
    f.write("### Head Dictionary ###\n")
    f.write("format: (key, class)\n")
    f.write("------------------------\n")
    for item in head_dict.items():
        f.write(str(item) + "\n")
