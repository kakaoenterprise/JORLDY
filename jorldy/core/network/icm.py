import torch
import torch.nn.functional as F

from .rnd import normalize_obs
from .utils import RewardForwardFilter, RunningMeanStd


def define_mlp_head_weight(instance, D_in, D_hidden, feature_size):
    instance.fc1 = torch.nn.Linear(D_in, D_hidden)
    instance.fc2 = torch.nn.Linear(D_hidden, feature_size)


def define_mlp_batch_norm(instance, D_hidden, feature_size):
    instance.bn1 = torch.nn.BatchNorm1d(D_hidden)
    instance.bn2 = torch.nn.BatchNorm1d(feature_size)

    instance.bn1_next = torch.nn.BatchNorm1d(D_hidden)


def mlp_head(instance, s, s_next):
    if instance.batch_norm:
        s = F.elu(instance.bn1(instance.fc1(s)))
        s = F.elu(instance.bn2(instance.fc2(s)))

        s_next = F.elu(instance.bn1_next(instance.fc1(s_next)))
    else:
        s = F.elu(instance.fc1(s))
        s = F.elu(instance.fc2(s))

        s_next = F.elu(instance.fc1(s_next))

    s_next = F.elu(instance.fc2(s_next))

    return s, s_next


def define_conv_head_weight(instance, D_in):
    instance.conv1 = torch.nn.Conv2d(
        in_channels=D_in[0], out_channels=32, kernel_size=3, stride=2
    )
    instance.conv2 = torch.nn.Conv2d(
        in_channels=32, out_channels=32, kernel_size=3, stride=2
    )
    instance.conv3 = torch.nn.Conv2d(
        in_channels=32, out_channels=32, kernel_size=3, stride=2
    )
    instance.conv4 = torch.nn.Conv2d(
        in_channels=32, out_channels=32, kernel_size=3, stride=2
    )

    dim1 = ((D_in[1] - 3) // 2 + 1, (D_in[2] - 3) // 2 + 1)
    dim2 = ((dim1[0] - 3) // 2 + 1, (dim1[1] - 3) // 2 + 1)
    dim3 = ((dim2[0] - 3) // 2 + 1, (dim2[1] - 3) // 2 + 1)
    dim4 = ((dim3[0] - 3) // 2 + 1, (dim3[1] - 3) // 2 + 1)

    feature_size = 32 * dim4[0] * dim4[1]
    return feature_size


def define_conv_batch_norm(instance):
    instance.bn1_conv = torch.nn.BatchNorm2d(32)
    instance.bn2_conv = torch.nn.BatchNorm2d(32)
    instance.bn3_conv = torch.nn.BatchNorm2d(32)
    instance.bn4_conv = torch.nn.BatchNorm2d(32)

    instance.bn1_next_conv = torch.nn.BatchNorm2d(32)
    instance.bn2_next_conv = torch.nn.BatchNorm2d(32)
    instance.bn3_next_conv = torch.nn.BatchNorm2d(32)


def conv_head(instance, s, s_next):
    if instance.batch_norm:
        s = F.elu(instance.bn1_conv(instance.conv1(s)))
        s = F.elu(instance.bn2_conv(instance.conv2(s)))
        s = F.elu(instance.bn3_conv(instance.conv3(s)))
        s = F.elu(instance.bn4_conv(instance.conv4(s)))

        s_next = F.elu(instance.bn1_next_conv(instance.conv1(s_next)))
        s_next = F.elu(instance.bn2_next_conv(instance.conv2(s_next)))
        s_next = F.elu(instance.bn3_next_conv(instance.conv3(s_next)))

    else:
        s = F.elu(instance.conv1(s))
        s = F.elu(instance.conv2(s))
        s = F.elu(instance.conv3(s))
        s = F.elu(instance.conv4(s))

        s_next = F.elu(instance.conv1(s_next))
        s_next = F.elu(instance.conv2(s_next))
        s_next = F.elu(instance.conv3(s_next))

    s_next = F.elu(instance.conv4(s_next))
    s = s.view(s.size(0), -1)
    s_next = s_next.view(s_next.size(0), -1)

    return s, s_next


def define_forward_weight(instance, feature_size, D_hidden, D_out):
    if instance.action_type == "discrete":
        instance.forward_fc1 = torch.nn.Linear(feature_size + 1, D_hidden)
        instance.forward_fc2 = torch.nn.Linear(D_hidden + 1, feature_size)
    else:
        instance.forward_fc1 = torch.nn.Linear(feature_size + D_out, D_hidden)
        instance.forward_fc2 = torch.nn.Linear(D_hidden + D_out, feature_size)

    instance.forward_loss = torch.nn.MSELoss()


def define_inverse_weight(instance, feature_size, D_hidden, D_out):
    instance.inverse_fc1 = torch.nn.Linear(2 * feature_size, D_hidden)
    instance.inverse_fc2 = torch.nn.Linear(D_hidden, D_out)

    instance.inverse_loss = (
        torch.nn.CrossEntropyLoss()
        if instance.action_type == "discrete"
        else torch.nn.MSELoss()
    )


def forward_model(instance, s, a, s_next):
    x_forward = torch.cat((s, a), axis=1)
    x_forward = F.relu(instance.forward_fc1(x_forward))
    x_forward = torch.cat((x_forward, a), axis=1)
    x_forward = instance.forward_fc2(x_forward)

    l_f = instance.forward_loss(x_forward, s_next.detach())

    return x_forward, l_f


def inverse_model(instance, s, a, s_next):
    x_inverse = torch.cat((s, s_next), axis=1)
    x_inverse = F.relu(instance.inverse_fc1(x_inverse))
    x_inverse = instance.inverse_fc2(x_inverse)

    if instance.action_type == "discrete":
        l_i = instance.inverse_loss(x_inverse, a.view(-1).long())
    else:
        l_i = instance.inverse_loss(x_inverse, a)

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

        define_mlp_head_weight(self, D_in, D_hidden, feature_size)
        define_forward_weight(self, feature_size, D_hidden, D_out)
        define_inverse_weight(self, feature_size, D_hidden, D_out)

        if self.batch_norm:
            define_mlp_batch_norm(self, D_hidden, feature_size)

    def update_rms_obs(self, v):
        self.rms_obs.update(v)

    def update_rms_ri(self, v):
        self.rms_ri.update(v)

    def forward(self, s, a, s_next, update_ri=False):
        if self.obs_normalize:
            s = normalize_obs(s, self.rms_obs.mean, self.rms_obs.var)
            s_next = normalize_obs(s_next, self.rms_obs.mean, self.rms_obs.var)

        s, s_next = mlp_head(self, s, s_next)

        # Forward Model
        x_forward, l_f = forward_model(self, s, a, s_next)

        # Inverse Model
        l_i = inverse_model(self, s, a, s_next)

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

        feature_size = define_conv_head_weight(self, self.D_in)
        define_forward_weight(self, feature_size, D_hidden, D_out)
        define_inverse_weight(self, feature_size, D_hidden, D_out)

        if self.batch_norm:
            define_conv_batch_norm(self)

    def update_rms_obs(self, v):
        self.rms_obs.update(v / 255.0)

    def update_rms_ri(self, v):
        self.rms_ri.update(v)

    def forward(self, s, a, s_next, update_ri=False):
        if self.obs_normalize:
            s = normalize_obs(s, self.rms_obs.mean, self.rms_obs.var)
            s_next = normalize_obs(s_next, self.rms_obs.mean, self.rms_obs.var)

        s, s_next = conv_head(self, s, s_next)

        # Forward Model
        x_forward, l_f = forward_model(self, s, a, s_next)

        # Inverse Model
        l_i = inverse_model(self, s, a, s_next)

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

        feature_size_img = define_conv_head_weight(self, self.D_in_img)
        feature_size_mlp = 256

        define_mlp_head_weight(self, self.D_in_vec, D_hidden, feature_size_mlp)

        feature_size = feature_size_img + feature_size_mlp

        define_forward_weight(self, feature_size, D_hidden, D_out)
        define_inverse_weight(self, feature_size, D_hidden, D_out)

        if self.batch_norm:
            define_mlp_batch_norm(self, D_hidden, feature_size_mlp)
            define_conv_batch_norm(self)

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

        s_vec, s_next_vec = mlp_head(self, s_vec, s_next_vec)
        s_img, s_next_img = conv_head(self, s_img, s_next_img)

        s = torch.cat((s_img, s_vec), -1)
        s_next = torch.cat((s_next_img, s_next_vec), -1)

        # Forward Model
        x_forward, l_f = forward_model(self, s, a, s_next)

        # Inverse Model
        l_i = inverse_model(self, s, a, s_next)

        # Get Ri
        r_i = (self.eta * 0.5) * torch.sum(torch.abs(x_forward - s_next), axis=1)

        if update_ri:
            ri_update(r_i, self.num_workers, self.rff, self.update_rms_ri)

        if self.ri_normalize:
            r_i = r_i / (torch.sqrt(self.rms_ri.var) + 1e-7)

        return r_i, l_f, l_i
