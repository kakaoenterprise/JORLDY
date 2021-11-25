import torch
import torch.nn.functional as F

from .utils import RewardForwardFilter, RunningMeanStd

# normalize observation
# assumed state shape: (batch_size, dim_state)
def normalize_obs(obs, m, v):
    return torch.clip((obs - m) / (torch.sqrt(v) + 1e-7), min=-5.0, max=5.0)


def define_mlp_head_weight(instance, D_in, D_hidden, feature_size):
    instance.fc1_p_mlp = torch.nn.Linear(D_in, D_hidden)
    instance.fc2_p_mlp = torch.nn.Linear(D_hidden, feature_size)

    instance.fc1_t_mlp = torch.nn.Linear(D_in, D_hidden)
    instance.fc2_t_mlp = torch.nn.Linear(D_hidden, feature_size)


def define_mlp_batch_norm(instance, D_hidden, feature_size):
    instance.bn1_p_mlp = torch.nn.BatchNorm1d(D_hidden)
    instance.bn2_p_mlp = torch.nn.BatchNorm1d(feature_size)

    instance.bn1_t_mlp = torch.nn.BatchNorm1d(D_hidden)
    instance.bn2_t_mlp = torch.nn.BatchNorm1d(feature_size)


def mlp_head(instance, s_next):
    if instance.batch_norm:
        p = F.relu(instance.bn1_p_mlp(instance.fc1_p_mlp(s_next)))
        p = F.relu(instance.bn2_p_mlp(instance.fc2_p_mlp(p)))

        t = F.relu(instance.bn1_t_mlp(instance.fc1_t_mlp(s_next)))
        t = F.relu(instance.bn2_t_mlp(instance.fc2_t_mlp(t)))
    else:
        p = F.relu(instance.fc1_p_mlp(s_next))
        p = F.relu(instance.fc2_p_mlp(p))

        t = F.relu(instance.fc1_t_mlp(s_next))
        t = F.relu(instance.fc2_t_mlp(t))

    return p, t


def define_conv_head_weight(instance, D_in):
    dim1 = ((D_in[1] - 8) // 4 + 1, (D_in[2] - 8) // 4 + 1)
    dim2 = ((dim1[0] - 4) // 2 + 1, (dim1[1] - 4) // 2 + 1)
    dim3 = ((dim2[0] - 3) // 1 + 1, (dim2[1] - 3) // 1 + 1)

    feature_size = 64 * dim3[0] * dim3[1]

    # Predictor Networks
    instance.conv1_p = torch.nn.Conv2d(
        in_channels=D_in[0], out_channels=32, kernel_size=8, stride=4
    )
    instance.conv2_p = torch.nn.Conv2d(
        in_channels=32, out_channels=64, kernel_size=4, stride=2
    )
    instance.conv3_p = torch.nn.Conv2d(
        in_channels=64, out_channels=64, kernel_size=3, stride=1
    )

    # Target Networks
    instance.conv1_t = torch.nn.Conv2d(
        in_channels=D_in[0], out_channels=32, kernel_size=8, stride=4
    )
    instance.conv2_t = torch.nn.Conv2d(
        in_channels=32, out_channels=64, kernel_size=4, stride=2
    )
    instance.conv3_t = torch.nn.Conv2d(
        in_channels=64, out_channels=64, kernel_size=3, stride=1
    )

    return feature_size


def define_conv_batch_norm(instance):
    instance.bn1_p_conv = torch.nn.BatchNorm2d(32)
    instance.bn2_p_conv = torch.nn.BatchNorm2d(64)
    instance.bn3_p_conv = torch.nn.BatchNorm2d(64)

    instance.bn1_t_conv = torch.nn.BatchNorm2d(32)
    instance.bn2_t_conv = torch.nn.BatchNorm2d(64)
    instance.bn3_t_conv = torch.nn.BatchNorm2d(64)


def conv_head(instance, s_next):
    if instance.batch_norm:
        p = F.relu(instance.bn1_p_conv(instance.conv1_p(s_next)))
        p = F.relu(instance.bn2_p_conv(instance.conv2_p(p)))
        p = F.relu(instance.bn3_p_conv(instance.conv3_p(p)))

        t = F.relu(instance.bn1_t_conv(instance.conv1_t(s_next)))
        t = F.relu(instance.bn2_t_conv(instance.conv2_t(t)))
        t = F.relu(instance.bn3_t_conv(instance.conv3_t(t)))
    else:
        p = F.relu(instance.conv1_p(s_next))
        p = F.relu(instance.conv2_p(p))
        p = F.relu(instance.conv3_p(p))

        t = F.relu(instance.conv1_t(s_next))
        t = F.relu(instance.conv2_t(t))
        t = F.relu(instance.conv3_t(t))

    p = p.view(p.size(0), -1)
    t = t.view(t.size(0), -1)

    return p, t


def define_fc_layers_weight(instance, feature_size, D_hidden):
    instance.fc1_p = torch.nn.Linear(feature_size, D_hidden)
    instance.fc2_p = torch.nn.Linear(D_hidden, D_hidden)
    instance.fc3_p = torch.nn.Linear(D_hidden, D_hidden)

    instance.fc1_t = torch.nn.Linear(feature_size, D_hidden)


def fc_layers(instance, p, t):
    p = F.relu(instance.fc1_p(p))
    p = F.relu(instance.fc2_p(p))
    p = instance.fc3_p(p)

    t = instance.fc1_t(t)

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

        define_mlp_head_weight(self, D_in, D_hidden, feature_size)

        if self.batch_norm:
            define_mlp_batch_norm(self, D_hidden, feature_size)

    def update_rms_obs(self, v):
        self.rms_obs.update(v)

    def update_rms_ri(self, v):
        self.rms_ri.update(v)

    def forward(self, s_next, update_ri=False):
        if self.obs_normalize:
            s_next = normalize_obs(s_next, self.rms_obs.mean, self.rms_obs.var)

        p, t = mlp_head(self, s_next)

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

        feature_size = define_conv_head_weight(self, D_in)
        define_conv_batch_norm(self)
        define_fc_layers_weight(self, feature_size, D_hidden)

    def update_rms_obs(self, v):
        self.rms_obs.update(v / 255.0)

    def update_rms_ri(self, v):
        self.rms_ri.update(v)

    def forward(self, s_next, update_ri=False):
        s_next = s_next / 255.0
        if self.obs_normalize:
            s_next = normalize_obs(s_next, self.rms_obs.mean, self.rms_obs.var)

        p, t = conv_head(self, s_next)
        p, t = fc_layers(self, p, t)

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

        feature_size_img = define_conv_head_weight(self, self.D_in_img)
        define_conv_batch_norm(self)

        feature_size_mlp = 256
        define_mlp_head_weight(self, self.D_in_vec, D_hidden, feature_size_mlp)

        define_mlp_batch_norm(self, D_hidden, feature_size_mlp)

        feature_size = feature_size_img + feature_size_mlp

        define_fc_layers_weight(self, feature_size, D_hidden)

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

        p_conv, t_conv = conv_head(self, s_next_img)
        p_mlp, t_mlp = mlp_head(self, s_next_vec)

        p = torch.cat((p_conv, p_mlp), -1)
        t = torch.cat((t_conv, t_mlp), -1)

        p, t = fc_layers(self, p, t)

        r_i = torch.mean(torch.square(p - t), axis=1)

        if update_ri:
            ri_update(r_i, self.num_workers, self.rff, self.update_rms_ri)

        if self.ri_normalize:
            r_i = r_i / (torch.sqrt(self.rms_ri.var) + 1e-7)

        return r_i
