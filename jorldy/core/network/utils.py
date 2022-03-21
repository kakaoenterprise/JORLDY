import torch
import torch.nn.functional as F

# codes from https://github.com/openai/random-network-distillation
class RewardForwardFilter(torch.nn.Module):
    def __init__(self, gamma, num_workers):
        super(RewardForwardFilter, self).__init__()
        self.rewems = torch.nn.Parameter(torch.zeros(num_workers), requires_grad=False)
        self.gamma = gamma

    def update(self, rews):
        self.rewems.data = self.rewems * self.gamma + rews
        return self.rewems


# codes modified from https://github.com/openai/random-network-distillation
class RunningMeanStd(torch.nn.Module):
    def __init__(self, shape, epsilon=1e-4):
        super(RunningMeanStd, self).__init__()

        self.mean = torch.nn.Parameter(torch.zeros(shape), requires_grad=False)
        self.var = torch.nn.Parameter(torch.zeros(shape), requires_grad=False)
        self.count = torch.nn.Parameter(torch.tensor(epsilon), requires_grad=False)

    def update(self, x):
        batch_mean, batch_std, batch_count = x.mean(axis=0), x.std(axis=0), x.shape[0]
        batch_var = torch.square(batch_std)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * (batch_count)
        M2 = (
            m_a
            + m_b
            + torch.square(delta)
            * self.count
            * batch_count
            / (self.count + batch_count)
        )
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean.data = new_mean
        self.var.data = new_var
        self.count.data = new_count


def noisy_l(x, mu_w, sig_w, mu_b, sig_b, noise_type, is_train):
    if noise_type == "factorized":
        # Factorized Gaussian Noise
        if is_train:
            eps_i = torch.randn(mu_w.size()[0]).to(x.device)
            eps_j = torch.randn(mu_b.size()[0]).to(x.device)

            f_eps_i = torch.sign(eps_i) * torch.sqrt(torch.abs(eps_i))
            f_eps_j = torch.sign(eps_j) * torch.sqrt(torch.abs(eps_j))

            eps_w = torch.matmul(
                torch.unsqueeze(f_eps_i, 1), torch.unsqueeze(f_eps_j, 0)
            )
            eps_b = f_eps_j
        else:
            eps_w = torch.zeros(mu_w.size()[0], mu_b.size()[0]).to(x.device)
            eps_b = torch.zeros(1, mu_b.size()[0]).to(x.device)
    else:
        # Independent Gaussian Noise
        if is_train:
            eps_w = torch.randn(mu_w.size()).to(x.device)
            eps_b = torch.randn(mu_b.size()).to(x.device)
        else:
            eps_w = torch.zeros(mu_w.size()).to(x.device)
            eps_b = torch.zeros(mu_b.size()).to(x.device)

    weight = mu_w + sig_w * eps_w
    bias = mu_b + sig_b * eps_b

    y = torch.matmul(x, weight) + bias

    return y


def init_weights(shape, noise_type):
    if noise_type == "factorized":
        mu_init = 1.0 / (shape[0] ** 0.5)
        sig_init = 0.5 / (shape[0] ** 0.5)
    else:
        mu_init = (3.0 / shape[0]) ** 0.5
        sig_init = 0.017

    mu_w = torch.nn.Parameter(torch.empty(shape))
    sig_w = torch.nn.Parameter(torch.empty(shape))
    mu_b = torch.nn.Parameter(torch.empty(shape[1]))
    sig_b = torch.nn.Parameter(torch.empty(shape[1]))

    mu_w.data.uniform_(-mu_init, mu_init)
    mu_b.data.uniform_(-mu_init, mu_init)
    sig_w.data.uniform_(sig_init, sig_init)
    sig_b.data.uniform_(sig_init, sig_init)

    return mu_w, sig_w, mu_b, sig_b


def orthogonal_init(layer, nonlinearity="relu"):
    if isinstance(nonlinearity, str):
        if nonlinearity == "policy":
            gain = 0.01
        else:
            gain = torch.nn.init.calculate_gain(nonlinearity)
    else:  # consider nonlinearity is gain
        gain = nonlinearity

    if isinstance(layer, list):
        for l in layer:
            torch.nn.init.orthogonal_(l.weight.data, gain)
            torch.nn.init.zeros_(l.bias.data)
    else:
        torch.nn.init.orthogonal_(layer.weight.data, gain)
        torch.nn.init.zeros_(layer.bias.data)


class Converter:
    def __init__(self, support):
        self.support = support

    # codes modified from https://github.com/werner-duvaud/muzero-general
    def vector2scalar(self, prob):
        """prediction value & dynamics reward output(vector:distribution) -> output(scalar:value)"""
        # get supports
        support = (
            torch.tensor([x for x in range(-self.support, self.support+1)])
            .expand(prob.shape)
            .float()
            .to(device=prob.device)
        )

        # convert to scalar
        scalar = torch.sum(support * prob, dim=-1, keepdim=True)

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
    def scalar2vector(self, scalar):
        """initiate target distribution from scalar(batch-2D) & project to learn batch-data"""
        # reduce scale
        scalar = (
                torch.sign(scalar) * (torch.sqrt(torch.abs(scalar) + 1) - 1) + 0.001 * scalar
        )
        scalar = scalar.view(scalar.shape)
        scalar = torch.clamp(scalar, -self.support, self.support+1)

        # target distribution projection(distribute probability for lower support)
        floor = scalar.floor()
        prob = scalar - floor
        dist = torch.zeros(
            scalar.shape[0], scalar.shape[1], (self.support << 1) + 1
        ).to(scalar.device)
        dist.scatter_(
            2, (floor + self.support).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
        )

        # target distribution projection(distribute probability for higher support)
        idx = floor + self.support + 1
        prob = prob.masked_fill_((self.support << 1) < idx, 0.0)
        idx = idx.masked_fill_((self.support << 1) < idx, 0.0)
        dist.scatter_(2, idx.long().unsqueeze(-1), prob.unsqueeze(-1))

        return dist
