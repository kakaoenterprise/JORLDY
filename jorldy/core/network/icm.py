import torch
import torch.nn.functional as F

from .rnd import *


def mlp_head_weight(D_in, D_hidden, feature_size):
    fc1 = torch.nn.Linear(D_in, D_hidden)
    fc2 = torch.nn.Linear(D_hidden, feature_size)

    return fc1, fc2


def mlp_batch_norm(D_hidden, feature_size):
    bn1 = torch.nn.BatchNorm1d(D_hidden)
    bn2 = torch.nn.BatchNorm1d(feature_size)

    bn1_next = torch.nn.BatchNorm1d(D_hidden)

    return bn1, bn2, bn1_next


def mlp_head(s, s_next, batch_norm, fc1, fc2, bn1, bn2, bn1_next):
    if batch_norm:
        s = F.elu(bn1(fc1(s)))
        s = F.elu(bn2(fc2(s)))

        s_next = F.elu(bn1_next(fc1(s_next)))
    else:
        s = F.elu(fc1(s))
        s = F.elu(fc2(s))

        s_next = F.elu(fc1(s_next))

    s_next = F.elu(fc2(s_next))

    return s, s_next


def conv_head_weight(D_in):
    conv1 = torch.nn.Conv2d(
        in_channels=D_in[0], out_channels=32, kernel_size=3, stride=2
    )
    conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
    conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
    conv4 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)

    dim1 = ((D_in[1] - 3) // 2 + 1, (D_in[2] - 3) // 2 + 1)
    dim2 = ((dim1[0] - 3) // 2 + 1, (dim1[1] - 3) // 2 + 1)
    dim3 = ((dim2[0] - 3) // 2 + 1, (dim2[1] - 3) // 2 + 1)
    dim4 = ((dim3[0] - 3) // 2 + 1, (dim3[1] - 3) // 2 + 1)

    feature_size = 32 * dim4[0] * dim4[1]

    return conv1, conv2, conv3, conv4, feature_size


def conv_batch_norm():
    bn1 = torch.nn.BatchNorm2d(32)
    bn2 = torch.nn.BatchNorm2d(32)
    bn3 = torch.nn.BatchNorm2d(32)
    bn4 = torch.nn.BatchNorm2d(32)

    bn1_next = torch.nn.BatchNorm2d(32)
    bn2_next = torch.nn.BatchNorm2d(32)
    bn3_next = torch.nn.BatchNorm2d(32)

    return bn1, bn2, bn3, bn4, bn1_next, bn2_next, bn3_next


def conv_head(
    s,
    s_next,
    batch_norm,
    conv1,
    conv2,
    conv3,
    conv4,
    bn1,
    bn2,
    bn3,
    bn4,
    bn1_next,
    bn2_next,
    bn3_next,
):
    if batch_norm:
        s = F.elu(bn1(conv1(s)))
        s = F.elu(bn2(conv2(s)))
        s = F.elu(bn3(conv3(s)))
        s = F.elu(bn4(conv4(s)))

        s_next = F.elu(bn1_next(conv1(s_next)))
        s_next = F.elu(bn2_next(conv2(s_next)))
        s_next = F.elu(bn3_next(conv3(s_next)))

    else:
        s = F.elu(conv1(s))
        s = F.elu(conv2(s))
        s = F.elu(conv3(s))
        s = F.elu(conv4(s))

        s_next = F.elu(conv1(s_next))
        s_next = F.elu(conv2(s_next))
        s_next = F.elu(conv3(s_next))

    s_next = F.elu(conv4(s_next))
    s = s.view(s.size(0), -1)
    s_next = s_next.view(s_next.size(0), -1)

    return s, s_next


def forward_weight(feature_size, D_hidden, D_out, action_type):
    if action_type == "discrete":
        forward_fc1 = torch.nn.Linear(feature_size + 1, D_hidden)
        forward_fc2 = torch.nn.Linear(D_hidden + 1, feature_size)
    else:
        forward_fc1 = torch.nn.Linear(feature_size + D_out, D_hidden)
        forward_fc2 = torch.nn.Linear(D_hidden + D_out, feature_size)

    forward_loss = torch.nn.MSELoss()

    return forward_fc1, forward_fc2, forward_loss


def inverse_weight(feature_size, D_hidden, D_out, action_type):
    inverse_fc1 = torch.nn.Linear(2 * feature_size, D_hidden)
    inverse_fc2 = torch.nn.Linear(D_hidden, D_out)

    inverse_loss = (
        torch.nn.CrossEntropyLoss() if action_type == "discrete" else torch.nn.MSELoss()
    )

    return inverse_fc1, inverse_fc2, inverse_loss


def forward_model(s, a, s_next, forward_loss, forward_fc1, forward_fc2):
    x_forward = torch.cat((s, a), axis=1)
    x_forward = F.relu(forward_fc1(x_forward))
    x_forward = torch.cat((x_forward, a), axis=1)
    x_forward = forward_fc2(x_forward)

    l_f = forward_loss(x_forward, s_next.detach())

    return x_forward, l_f


def inverse_model(s, a, s_next, action_type, inverse_loss, inverse_fc1, inverse_fc2):
    x_inverse = torch.cat((s, s_next), axis=1)
    x_inverse = F.relu(inverse_fc1(x_inverse))
    x_inverse = inverse_fc2(x_inverse)

    if action_type == "discrete":
        l_i = inverse_loss(x_inverse, a.view(-1).long())
    else:
        l_i = inverse_loss(x_inverse, a)

    return l_i


def ri_update(r_i, num_workers, rff, update_rms_ri):
    ri_T = r_i.view(num_workers, -1).T  # (n_batch, n_workers)
    rewems = torch.stack(
        [rff.update(rit.detach()) for rit in ri_T]
    ).ravel()  # (n_batch, n_workers) -> (n_batch * n_workers)
    update_rms_ri(rewems)


class ICM_MLP(torch.nn.Module):
    def __init__(
        self,
        D_in,
        D_out,
        num_workers,
        gamma,
        eta,
        action_type,
        ri_normalize=True,
        obs_normalize=True,
        batch_norm=True,
        D_hidden=256,
    ):
        super(ICM_MLP, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.num_workers = num_workers
        self.eta = eta
        self.action_type = action_type

        self.rms_obs = RunningMeanStd(D_in)
        self.rms_ri = RunningMeanStd(1)
        self.rff = RewardForwardFilter(gamma, num_workers)
        self.obs_normalize = obs_normalize
        self.ri_normalize = ri_normalize
        self.batch_norm = batch_norm

        feature_size = 256

        self.fc1, self.fc2 = mlp_head_weight(D_in, D_hidden, feature_size)
        self.forward_fc1, self.forward_fc2, self.forward_loss = forward_weight(
            feature_size, D_hidden, D_out, action_type
        )
        self.inverse_fc1, self.inverse_fc2, self.inverse_loss = inverse_weight(
            feature_size, D_hidden, D_out, action_type
        )

        self.bn1, self.bn2, self.bn1_next = mlp_batch_norm(D_hidden, feature_size)

    def update_rms_obs(self, v):
        self.rms_obs.update(v)

    def update_rms_ri(self, v):
        self.rms_ri.update(v)

    def forward(self, s, a, s_next, update_ri=False):
        if self.obs_normalize:
            s = normalize_obs(s, self.rms_obs.mean, self.rms_obs.var)
            s_next = normalize_obs(s_next, self.rms_obs.mean, self.rms_obs.var)

        s, s_next = mlp_head(
            s,
            s_next,
            self.batch_norm,
            self.fc1,
            self.fc2,
            self.bn1,
            self.bn2,
            self.bn1_next,
        )

        # Forward Model
        x_forward, l_f = forward_model(
            s, a, s_next, self.forward_loss, self.forward_fc1, self.forward_fc2
        )

        # Inverse Model
        l_i = inverse_model(
            s,
            a,
            s_next,
            self.action_type,
            self.inverse_loss,
            self.inverse_fc1,
            self.inverse_fc2,
        )

        # Get Ri
        r_i = (self.eta * 0.5) * torch.sum(torch.abs(x_forward - s_next), axis=1)

        if update_ri:
            ri_update(r_i, self.num_workers, self.rff, self.update_rms_ri)

        if self.ri_normalize:
            r_i = r_i / (torch.sqrt(self.rms_ri.var) + 1e-7)

        return r_i, l_f, l_i


class ICM_CNN(torch.nn.Module):
    def __init__(
        self,
        D_in,
        D_out,
        num_workers,
        gamma,
        eta,
        action_type,
        ri_normalize=True,
        obs_normalize=True,
        batch_norm=True,
        D_hidden=256,
    ):
        super(ICM_CNN, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.num_workers = num_workers
        self.eta = eta
        self.action_type = action_type

        self.rms_obs = RunningMeanStd(D_in)
        self.rms_ri = RunningMeanStd(1)
        self.rff = RewardForwardFilter(gamma, num_workers)
        self.obs_normalize = obs_normalize
        self.ri_normalize = ri_normalize
        self.batch_norm = batch_norm

        self.conv1, self.conv2, self.conv3, self.conv4, feature_size = conv_head_weight(
            self.D_in
        )
        self.forward_fc1, self.forward_fc2, self.forward_loss = forward_weight(
            feature_size, D_hidden, D_out, action_type
        )
        self.inverse_fc1, self.inverse_fc2, self.inverse_loss = inverse_weight(
            feature_size, D_hidden, D_out, action_type
        )

        if self.batch_norm:
            (
                self.bn1,
                self.bn2,
                self.bn3,
                self.bn4,
                self.bn1_next,
                self.bn2_next,
                self.bn3_next,
            ) = conv_batch_norm()

    def update_rms_obs(self, v):
        self.rms_obs.update(v / 255.0)

    def update_rms_ri(self, v):
        self.rms_ri.update(v)

    def forward(self, s, a, s_next, update_ri=False):
        if self.obs_normalize:
            s = normalize_obs(s, self.rms_obs.mean, self.rms_obs.var)
            s_next = normalize_obs(s_next, self.rms_obs.mean, self.rms_obs.var)

        s, s_next = conv_head(
            s,
            s_next,
            self.batch_norm,
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.bn1,
            self.bn2,
            self.bn3,
            self.bn4,
            self.bn1_next,
            self.bn2_next,
            self.bn3_next,
        )

        # Forward Model
        x_forward, l_f = forward_model(
            s, a, s_next, self.forward_loss, self.forward_fc1, self.forward_fc2
        )

        # Inverse Model
        l_i = inverse_model(
            s,
            a,
            s_next,
            self.action_type,
            self.inverse_loss,
            self.inverse_fc1,
            self.inverse_fc2,
        )

        # Get Ri
        r_i = (self.eta * 0.5) * torch.sum(torch.abs(x_forward - s_next), axis=1)

        if update_ri:
            ri_update(r_i, self.num_workers, self.rff, self.update_rms_ri)

        if self.ri_normalize:
            r_i = r_i / (torch.sqrt(self.rms_ri.var) + 1e-7)

        return r_i, l_f, l_i


class ICM_Multi(torch.nn.Module):
    def __init__(
        self,
        D_in,
        D_out,
        num_workers,
        gamma,
        eta,
        action_type,
        ri_normalize=True,
        obs_normalize=True,
        batch_norm=True,
        D_hidden=256,
    ):
        super(ICM_Multi, self).__init__()
        self.D_in_img = D_in[0]
        self.D_in_vec = D_in[1]

        self.D_out = D_out
        self.num_workers = num_workers
        self.eta = eta
        self.action_type = action_type

        self.rms_obs_img = RunningMeanStd(self.D_in_img)
        self.rms_obs_vec = RunningMeanStd(self.D_in_vec)

        self.rms_ri = RunningMeanStd(1)
        self.rff = RewardForwardFilter(gamma, num_workers)
        self.obs_normalize = obs_normalize
        self.ri_normalize = ri_normalize
        self.batch_norm = batch_norm

        (
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            feature_size_img,
        ) = conv_head_weight(self.D_in_img)

        feature_size_mlp = 256

        self.fc1, self.fc2 = mlp_head_weight(self.D_in_vec, D_hidden, feature_size_mlp)

        feature_size = feature_size_img + feature_size_mlp

        self.forward_fc1, self.forward_fc2, self.forward_loss = forward_weight(
            feature_size, D_hidden, D_out, action_type
        )
        self.inverse_fc1, self.inverse_fc2, self.inverse_loss = inverse_weight(
            feature_size, D_hidden, D_out, action_type
        )

        if self.batch_norm:
            self.bn1_mlp, self.bn2_mlp, self.bn1_next_mlp = mlp_batch_norm(
                D_hidden, feature_size_mlp
            )
            (
                self.bn1_conv,
                self.bn2_conv,
                self.bn3_conv,
                self.bn4_conv,
                self.bn1_next_conv,
                self.bn2_next_conv,
                self.bn3_next_conv,
            ) = conv_batch_norm()

    def update_rms_obs(self, v):
        self.rms_obs_img.update(v[0] / 255.0)
        self.rms_obs_vec.update(v[1])

    def update_rms_ri(self, v):
        self.rms_ri.update(v)

    def forward(self, s, a, s_next, update_ri=False):
        s_img = s[0]
        s_vec = s[1]

        s_next_img = s_next[0]
        s_next_vec = s_next[1]

        if self.obs_normalize:
            s_img = normalize_obs(s_img, self.rms_obs_img.mean, self.rms_obs_img.var)
            s_vec = normalize_obs(s_vec, self.rms_obs_vec.mean, self.rms_obs_vec.var)
            s_next_img = normalize_obs(
                s_next_img, self.rms_obs_img.mean, self.rms_obs_img.var
            )
            s_next_vec = normalize_obs(
                s_next_vec, self.rms_obs_vec.mean, self.rms_obs_vec.var
            )

        s_vec, s_next_vec = mlp_head(
            s_vec,
            s_next_vec,
            self.batch_norm,
            self.fc1,
            self.fc2,
            self.bn1_mlp,
            self.bn2_mlp,
            self.bn1_next_mlp,
        )
        s_img, s_next_img = conv_head(
            s_img,
            s_next_img,
            self.batch_norm,
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.bn1_conv,
            self.bn2_conv,
            self.bn3_conv,
            self.bn4_conv,
            self.bn1_next_conv,
            self.bn2_next_conv,
            self.bn3_next_conv,
        )

        s = torch.cat((s_img, s_vec), -1)
        s_next = torch.cat((s_next_img, s_next_vec), -1)

        # Forward Model
        x_forward, l_f = forward_model(
            s, a, s_next, self.forward_loss, self.forward_fc1, self.forward_fc2
        )

        # Inverse Model
        l_i = inverse_model(
            s,
            a,
            s_next,
            self.action_type,
            self.inverse_loss,
            self.inverse_fc1,
            self.inverse_fc2,
        )

        # Get Ri
        r_i = (self.eta * 0.5) * torch.sum(torch.abs(x_forward - s_next), axis=1)

        if update_ri:
            ri_update(r_i, self.num_workers, self.rff, self.update_rms_ri)

        if self.ri_normalize:
            r_i = r_i / (torch.sqrt(self.rms_ri.var) + 1e-7)

        return r_i, l_f, l_i
