import torch
import torch.nn.functional as F

from .utils import RewardForwardFilter, RunningMeanStd

# normalize observation
# assumed state shape: (batch_size, dim_state)
def normalize_obs(obs, m, v):
    return torch.clip((obs - m) / (torch.sqrt(v) + 1e-7), min=-5.0, max=5.0)


def mlp_head_weight(D_in, D_hidden, feature_size):
    fc1_p = torch.nn.Linear(D_in, D_hidden)
    fc2_p = torch.nn.Linear(D_hidden, feature_size)

    fc1_t = torch.nn.Linear(D_in, D_hidden)
    fc2_t = torch.nn.Linear(D_hidden, feature_size)

    return fc1_p, fc2_p, fc1_t, fc2_t


def mlp_batch_norm(D_hidden, feature_size):
    bn1_p = torch.nn.BatchNorm1d(D_hidden)
    bn2_p = torch.nn.BatchNorm1d(feature_size)

    bn1_t = torch.nn.BatchNorm1d(D_hidden)
    bn2_t = torch.nn.BatchNorm1d(feature_size)

    return bn1_p, bn2_p, bn1_t, bn2_t


def mlp_head(
    s_next, batch_norm, fc1_p, fc2_p, fc1_t, fc2_t, bn1_p, bn2_p, bn1_t, bn2_t
):
    if batch_norm:
        p = F.relu(bn1_p(fc1_p(s_next)))
        p = F.relu(bn2_p(fc2_p(p)))

        t = F.relu(bn1_t(fc1_t(s_next)))
        t = F.relu(bn2_t(fc2_t(t)))
    else:
        p = F.relu(fc1_p(s_next))
        p = F.relu(fc2_p(p))

        t = F.relu(fc1_t(s_next))
        t = F.relu(fc2_t(t))

    return p, t


def conv_head_weight(D_in):
    dim1 = ((D_in[1] - 8) // 4 + 1, (D_in[2] - 8) // 4 + 1)
    dim2 = ((dim1[0] - 4) // 2 + 1, (dim1[1] - 4) // 2 + 1)
    dim3 = ((dim2[0] - 3) // 1 + 1, (dim2[1] - 3) // 1 + 1)

    feature_size = 64 * dim3[0] * dim3[1]

    # Predictor Networks
    conv1_p = torch.nn.Conv2d(
        in_channels=D_in[0], out_channels=32, kernel_size=8, stride=4
    )
    conv2_p = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
    conv3_p = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

    # Target Networks
    conv1_t = torch.nn.Conv2d(
        in_channels=D_in[0], out_channels=32, kernel_size=8, stride=4
    )
    conv2_t = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
    conv3_t = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

    return conv1_p, conv2_p, conv3_p, conv1_t, conv2_t, conv3_t, feature_size


def conv_batch_norm():
    bn1_p = torch.nn.BatchNorm2d(32)
    bn2_p = torch.nn.BatchNorm2d(64)
    bn3_p = torch.nn.BatchNorm2d(64)

    bn1_t = torch.nn.BatchNorm2d(32)
    bn2_t = torch.nn.BatchNorm2d(64)
    bn3_t = torch.nn.BatchNorm2d(64)

    return bn1_p, bn2_p, bn3_p, bn1_t, bn2_t, bn3_t


def conv_head(
    s_next,
    batch_norm,
    conv1_p,
    conv2_p,
    conv3_p,
    conv1_t,
    conv2_t,
    conv3_t,
    bn1_p,
    bn2_p,
    bn3_p,
    bn1_t,
    bn2_t,
    bn3_t,
):
    if batch_norm:
        p = F.relu(bn1_p(conv1_p(s_next)))
        p = F.relu(bn2_p(conv2_p(p)))
        p = F.relu(bn3_p(conv3_p(p)))

        t = F.relu(bn1_t(conv1_t(s_next)))
        t = F.relu(bn2_t(conv2_t(t)))
        t = F.relu(bn3_t(conv3_t(t)))
    else:
        p = F.relu(conv1_p(s_next))
        p = F.relu(conv2_p(p))
        p = F.relu(conv3_p(p))

        t = F.relu(conv1_t(s_next))
        t = F.relu(conv2_t(t))
        t = F.relu(conv3_t(t))

    p = p.view(p.size(0), -1)
    t = t.view(t.size(0), -1)

    return p, t


def fc_layers_weight(feature_size, D_hidden):
    fc1_p = torch.nn.Linear(feature_size, D_hidden)
    fc2_p = torch.nn.Linear(D_hidden, D_hidden)
    fc3_p = torch.nn.Linear(D_hidden, D_hidden)

    fc1_t = torch.nn.Linear(feature_size, D_hidden)

    return fc1_p, fc2_p, fc3_p, fc1_t


def fc_layers(p, t, fc1_p, fc2_p, fc3_p, fc1_t):
    p = F.relu(fc1_p(p))
    p = F.relu(fc2_p(p))
    p = fc3_p(p)

    t = fc1_t(t)

    return p, t


def ri_update(r_i, num_workers, rff, update_rms_ri):
    ri_T = r_i.view(num_workers, -1).T  # (n_batch, n_workers)
    rewems = torch.stack(
        [rff.update(rit.detach()) for rit in ri_T]
    ).ravel()  # (n_batch, n_workers) -> (n_batch * n_workers)
    update_rms_ri(rewems)


class RND_MLP(torch.nn.Module):
    def __init__(
        self,
        D_in,
        D_out,
        num_workers,
        gamma_i,
        ri_normalize=True,
        obs_normalize=True,
        batch_norm=True,
        D_hidden=256,
    ):
        super(RND_MLP, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.num_workers = num_workers
        self.rms_obs = RunningMeanStd(D_in)
        self.rms_ri = RunningMeanStd(1)
        self.rff = RewardForwardFilter(gamma_i, num_workers)
        self.obs_normalize = obs_normalize
        self.ri_normalize = ri_normalize
        self.batch_norm = batch_norm

        feature_size = 256

        self.fc1_p, self.fc2_p, self.fc1_t, self.fc2_t = mlp_head_weight(
            D_in, D_hidden, feature_size
        )
        self.bn1_p, self.bn2_p, self.bn1_t, self.bn2_t = mlp_batch_norm(
            D_hidden, feature_size
        )

    def update_rms_obs(self, v):
        self.rms_obs.update(v)

    def update_rms_ri(self, v):
        self.rms_ri.update(v)

    def forward(self, s_next, update_ri=False):
        if self.obs_normalize:
            s_next = normalize_obs(s_next, self.rms_obs.mean, self.rms_obs.var)

        p, t = mlp_head(
            s_next,
            self.batch_norm,
            self.fc1_p,
            self.fc2_p,
            self.fc1_t,
            self.fc2_t,
            self.bn1_p,
            self.bn2_p,
            self.bn1_t,
            self.bn2_t,
        )

        r_i = torch.mean(torch.square(p - t), axis=1)

        if update_ri:
            ri_update(r_i, self.num_workers, self.rff, self.update_rms_ri)

        if self.ri_normalize:
            r_i = r_i / (torch.sqrt(self.rms_ri.var) + 1e-7)

        return r_i


class RND_CNN(torch.nn.Module):
    def __init__(
        self,
        D_in,
        D_out,
        num_workers,
        gamma_i,
        ri_normalize=True,
        obs_normalize=True,
        batch_norm=True,
        D_hidden=512,
    ):
        super(RND_CNN, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.num_workers = num_workers
        self.rms_obs = RunningMeanStd(D_in)
        self.rms_ri = RunningMeanStd(1)
        self.rff = RewardForwardFilter(gamma_i, num_workers)
        self.obs_normalize = obs_normalize
        self.ri_normalize = ri_normalize
        self.batch_norm = batch_norm

        (
            self.conv1_p,
            self.conv2_p,
            self.conv3_p,
            self.conv1_t,
            self.conv2_t,
            self.conv3_t,
            feature_size,
        ) = conv_head_weight(D_in)
        (
            self.bn1_p,
            self.bn2_p,
            self.bn3_p,
            self.bn1_t,
            self.bn2_t,
            self.bn3_t,
        ) = conv_batch_norm()
        self.fc1_p, self.fc2_p, self.fc3_p, self.fc1_t = fc_layers_weight(
            feature_size, D_hidden
        )

    def update_rms_obs(self, v):
        self.rms_obs.update(v / 255.0)

    def update_rms_ri(self, v):
        self.rms_ri.update(v)

    def forward(self, s_next, update_ri=False):
        s_next = s_next / 255.0
        if self.obs_normalize:
            s_next = normalize_obs(s_next, self.rms_obs.mean, self.rms_obs.var)

        p, t = conv_head(
            s_next,
            self.batch_norm,
            self.conv1_p,
            self.conv2_p,
            self.conv3_p,
            self.conv1_t,
            self.conv2_t,
            self.conv3_t,
            self.bn1_p,
            self.bn2_p,
            self.bn3_p,
            self.bn1_t,
            self.bn2_t,
            self.bn3_t,
        )

        p, t = fc_layers(p, t, self.fc1_p, self.fc2_p, self.fc3_p, self.fc1_t)

        r_i = torch.mean(torch.square(p - t), axis=1)

        if update_ri:
            ri_update(r_i, self.num_workers, self.rff, self.update_rms_ri)

        if self.ri_normalize:
            r_i = r_i / (torch.sqrt(self.rms_ri.var) + 1e-7)

        return r_i


class RND_Multi(torch.nn.Module):
    def __init__(
        self,
        D_in,
        D_out,
        num_workers,
        gamma_i,
        ri_normalize=True,
        obs_normalize=True,
        batch_norm=True,
        D_hidden=512,
    ):
        super(RND_Multi, self).__init__()
        self.D_in_img = D_in[0]
        self.D_in_vec = D_in[1]

        self.D_out = D_out

        self.num_workers = num_workers
        self.rms_obs_img = RunningMeanStd(self.D_in_img)
        self.rms_obs_vec = RunningMeanStd(self.D_in_vec)

        self.rms_ri = RunningMeanStd(1)
        self.rff = RewardForwardFilter(gamma_i, num_workers)
        self.obs_normalize = obs_normalize
        self.ri_normalize = ri_normalize
        self.batch_norm = batch_norm

        (
            self.conv1_p,
            self.conv2_p,
            self.conv3_p,
            self.conv1_t,
            self.conv2_t,
            self.conv3_t,
            feature_size_img,
        ) = conv_head_weight(self.D_in_img)
        (
            self.bn1_p_conv,
            self.bn2_p_conv,
            self.bn3_p_conv,
            self.bn1_t_conv,
            self.bn2_t_conv,
            self.bn3_t_conv,
        ) = conv_batch_norm()

        feature_size_mlp = 256

        (
            self.fc1_p_mlp,
            self.fc2_p_mlp,
            self.fc1_t_mlp,
            self.fc2_t_mlp,
        ) = mlp_head_weight(self.D_in_vec, D_hidden, feature_size_mlp)

        self.bn1_p_mlp, self.bn2_p_mlp, self.bn1_t_mlp, self.bn2_t_mlp = mlp_batch_norm(
            D_hidden, feature_size_mlp
        )

        feature_size = feature_size_img + feature_size_mlp

        self.fc1_p, self.fc2_p, self.fc3_p, self.fc1_t = fc_layers_weight(
            feature_size, D_hidden
        )

    def update_rms_obs(self, v):
        self.rms_obs_img.update(v[0] / 255.0)
        self.rms_obs_vec.update(v[1])

    def update_rms_ri(self, v):
        self.rms_ri.update(v)

    def forward(self, s_next, update_ri=False):
        s_next_img = s_next[0]
        s_next_vec = s_next[1]

        s_next_img = s_next_img / 255.0

        if self.obs_normalize:
            s_next_img = normalize_obs(
                s_next_img, self.rms_obs_img.mean, self.rms_obs_img.var
            )
            s_next_vec = normalize_obs(
                s_next_vec, self.rms_obs_vec.mean, self.rms_obs_vec.var
            )

        p_conv, t_conv = conv_head(
            s_next_img,
            self.batch_norm,
            self.conv1_p,
            self.conv2_p,
            self.conv3_p,
            self.conv1_t,
            self.conv2_t,
            self.conv3_t,
            self.bn1_p_conv,
            self.bn2_p_conv,
            self.bn3_p_conv,
            self.bn1_t_conv,
            self.bn2_t_conv,
            self.bn3_t_conv,
        )

        p_mlp, t_mlp = mlp_head(
            s_next_vec,
            self.batch_norm,
            self.fc1_p_mlp,
            self.fc2_p_mlp,
            self.fc1_t_mlp,
            self.fc2_t_mlp,
            self.bn1_p_mlp,
            self.bn2_p_mlp,
            self.bn1_t_mlp,
            self.bn2_t_mlp,
        )

        p = torch.cat((p_conv, p_mlp), -1)
        t = torch.cat((t_conv, t_mlp), -1)

        p, t = fc_layers(p, t, self.fc1_p, self.fc2_p, self.fc3_p, self.fc1_t)

        r_i = torch.mean(torch.square(p - t), axis=1)

        if update_ri:
            ri_update(r_i, self.num_workers, self.rff, self.update_rms_ri)

        if self.ri_normalize:
            r_i = r_i / (torch.sqrt(self.rms_ri.var) + 1e-7)

        return r_i
