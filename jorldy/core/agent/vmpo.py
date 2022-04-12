import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from .reinforce import REINFORCE
from core.optimizer import Optimizer


class VMPO(REINFORCE):
    """Maximum A Posteriori Policy Optimization (MPO) agent.

    Args:
        optim_config (dict): dictionary of the optimizer info.
            (key: 'name', value: name of optimizer)
        batch_size (int): the number of samples in the one batch.
        n_step (int): The number of steps to run for each environment per update.
        n_epoch (int): Number of epoch when optimizing the surrogate.
        _lambda (float): Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        clip_grad_norm (float): gradient clipping threshold.
        min_eta (float): minimum value of eta.
        min_alpha_mu (float): minimum value of alpha_mu.
        min_alpha_sigma (float): minimum value of alpha_sigma.
        eps_eta (float): threshold of temperature loss term.
        eps_alpha_mu (float): threshold of mean part of Gaussian-KL constraint term.
        eps_alpha_sigma (float): threshold of variance part of Gaussian-KL constraint term.
        eta (float): Lagrange multipliers of temperature loss term.
        alpha_mu (float): Lagrange multipliers of mean part of Gaussian-KL constraint term (trust-region loss).
        alpha_sigma (float): Lagrange multipliers of variance part of Gaussian-KL constraint term.
    """

    def __init__(
        self,
        network="discrete_policy_value",
        optim_config={"name": "adam"},
        batch_size=32,
        n_step=128,
        n_epoch=1,
        _lambda=0.9,
        clip_grad_norm=1.0,
        # parameters unique to V-MPO
        min_eta=1e-8,
        min_alpha_mu=1e-8,
        min_alpha_sigma=1e-8,
        eps_eta=0.02,
        eps_alpha_mu=0.1,
        eps_alpha_sigma=0.1,
        eta=1.0,
        alpha_mu=1.0,
        alpha_sigma=1.0,
        **kwargs,
    ):
        super(VMPO, self).__init__(
            network=network,
            optim_config=optim_config,
            **kwargs,
        )

        self.batch_size = batch_size
        self.n_step = n_step
        self.n_epoch = n_epoch
        self._lambda = _lambda
        self.time_t = 0
        self.learn_stamp = 0
        self.clip_grad_norm = clip_grad_norm

        self.min_eta = torch.tensor(min_eta, device=self.device)
        self.min_alpha_mu = torch.tensor(min_alpha_mu, device=self.device)
        self.min_alpha_sigma = torch.tensor(min_alpha_sigma, device=self.device)

        self.eps_eta = eps_eta
        self.eps_alpha_mu = eps_alpha_mu
        self.eps_alpha_sigma = eps_alpha_sigma

        self.eta = torch.nn.Parameter(
            torch.tensor(eta, requires_grad=True).to(self.device)
        )
        self.alpha_mu = torch.nn.Parameter(
            torch.tensor(alpha_mu, requires_grad=True).to(self.device)
        )
        self.alpha_sigma = torch.nn.Parameter(
            torch.tensor(alpha_sigma, requires_grad=True).to(self.device)
        )

        self.reset_lgr_muls()

        self.optimizer = Optimizer(
            **optim_config,
            params=list(self.network.parameters())
            + [self.eta, self.alpha_mu, self.alpha_sigma],
        )

    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training)

        if self.action_type == "continuous":
            mu, std, _ = self.network(self.as_tensor(state))
            z = torch.normal(mu, std) if training else mu
            action = torch.tanh(z)
        else:
            pi, _ = self.network(self.as_tensor(state))
            action = (
                torch.multinomial(pi, 1)
                if training
                else torch.argmax(pi, dim=-1, keepdim=True)
            )
        return {"action": action.cpu().numpy()}

    def learn(self):
        transitions = self.memory.sample()
        for key in transitions.keys():
            transitions[key] = self.as_tensor(transitions[key])

        state = transitions["state"]
        action = transitions["action"]
        reward = transitions["reward"]
        next_state = transitions["next_state"]
        done = transitions["done"]

        # set advantage and log_pi_old
        with torch.no_grad():
            if self.action_type == "continuous":
                mu, std, value = self.network(state)
                m = Normal(mu, std)
                z = torch.atanh(torch.clamp(action, -1 + 1e-7, 1 - 1e-7))
                log_pi = m.log_prob(z)
                log_prob = log_pi.sum(axis=-1, keepdims=True)
                mu_old = mu
                std_old = std
            else:
                pi, value = self.network(state)
                pi_old = pi
                log_prob = torch.log(pi.gather(1, action.long()))
                log_pi_old = torch.log(pi)

            log_prob_old = log_prob

            next_value = self.network(next_state)[-1]
            delta = reward + (1 - done) * self.gamma * next_value - value
            adv = delta.clone()
            adv, done = adv.view(-1, self.n_step), done.view(-1, self.n_step)
            for t in reversed(range(self.n_step - 1)):
                adv[:, t] += (
                    (1 - done[:, t]) * self.gamma * self._lambda * adv[:, t + 1]
                )
            if self.use_standardization:
                adv = (adv - adv.mean(dim=1, keepdim=True)) / (
                    adv.std(dim=1, keepdim=True) + 1e-7
                )
            adv = adv.view(-1, 1)
            done = done.view(-1, 1)
            ret = adv + value

        # start train iteration
        actor_losses, critic_losses, eta_losses, alpha_losses = [], [], [], []
        idxs = np.arange(len(reward))
        for _ in range(self.n_epoch):
            np.random.shuffle(idxs)
            for offset in range(0, len(reward), self.batch_size):
                idx = idxs[offset : offset + self.batch_size]

                _state, _action, _ret, _next_state, _adv, _log_prob_old = map(
                    lambda x: [_x[idx] for _x in x] if isinstance(x, list) else x[idx],
                    [state, action, ret, next_state, adv, log_prob_old],
                )

                if self.action_type == "continuous":
                    _mu_old, _std_old = map(lambda x: x[idx], [mu_old, std_old])
                else:
                    _log_pi_old, _pi_old = map(lambda x: x[idx], [log_pi_old, pi_old])

                # select top 50% of advantages
                idx_tophalf = _adv > _adv.median()
                tophalf_adv = _adv[idx_tophalf]
                # calculate psi
                exp_adv_eta = torch.exp(tophalf_adv / self.eta)
                psi = exp_adv_eta / torch.sum(exp_adv_eta.detach())

                if self.action_type == "continuous":
                    mu, std, value = self.network(_state)
                    m = Normal(mu, std)
                    z = torch.atanh(torch.clamp(_action, -1 + 1e-7, 1 - 1e-7))
                    log_pi = m.log_prob(z)
                    log_prob = log_pi.sum(axis=-1, keepdims=True)
                else:
                    pi, value = self.network(_state)
                    log_prob = torch.log(pi.gather(1, _action.long()))
                    log_pi = torch.log(pi)

                critic_loss = F.mse_loss(value, _ret).mean()

                # calculate loss for eta
                eta_loss = self.eta * self.eps_eta + self.eta * torch.log(
                    torch.mean(exp_adv_eta)
                )

                # calculate policy loss (actor_loss)
                tophalf_log_prob = log_prob[idx_tophalf.squeeze(), :]
                actor_loss = -torch.sum(psi.detach().unsqueeze(1) * tophalf_log_prob)

                # calculate loss for alpha
                # NOTE: assumes that std are in the same shape as mu (hence vectors)
                #       hence each dimension of Gaussian distribution is independent
                if self.action_type == "continuous":
                    ss = 1.0 / (std**2)  # (batch_size * action_dim)
                    ss_old = 1.0 / (_std_old**2)  # (batch_size * action_dim)

                    # mu
                    d_mu = mu - _mu_old.detach()  # (batch_size * action_dim)
                    KLD_mu = 0.5 * torch.sum(
                        d_mu * 1.0 / ss_old.detach() * d_mu, axis=1
                    )
                    mu_loss = torch.mean(
                        self.alpha_mu * (self.eps_alpha_mu - KLD_mu.detach())
                        + self.alpha_mu.detach() * KLD_mu
                    )

                    # sigma
                    KLD_sigma = 0.5 * (
                        (
                            torch.sum(1.0 / ss * ss_old.detach(), axis=1)
                            - ss.shape[-1]
                            + torch.log(
                                torch.prod(ss, axis=1)
                                / torch.prod(ss_old.detach(), axis=1)
                            )
                        )
                    )
                    sigma_loss = torch.mean(
                        self.alpha_sigma * (self.eps_alpha_sigma - KLD_sigma.detach())
                        + self.alpha_sigma.detach() * KLD_sigma
                    )

                    alpha_loss = mu_loss + sigma_loss
                else:
                    KLD_pi = _pi_old.detach() * (_log_pi_old.detach() - log_pi)
                    KLD_pi = torch.sum(KLD_pi, axis=len(_pi_old.shape) - 1)
                    alpha_loss = torch.mean(
                        self.alpha_mu * (self.eps_alpha_mu - KLD_pi.detach())
                        + self.alpha_mu.detach() * KLD_pi
                    )

                loss = critic_loss + actor_loss + eta_loss + alpha_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.clip_grad_norm
                )
                self.optimizer.step()
                self.reset_lgr_muls()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                eta_losses.append(eta_loss.item())
                alpha_losses.append(alpha_loss.item())

        result = {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "eta_loss": np.mean(eta_losses),
            "alpha_loss": np.mean(alpha_losses),
            "eta": self.eta.item(),
            "alpha_mu": self.alpha_mu.item(),
            "alpha_sigma": self.alpha_sigma.item(),
        }
        return result

    # reset Lagrange multipliers: eta, alpha_{mu, sigma}
    def reset_lgr_muls(self):
        self.eta.data = torch.max(self.eta, self.min_eta)
        self.alpha_mu.data = torch.max(self.alpha_mu, self.min_alpha_mu)
        self.alpha_sigma.data = torch.max(self.alpha_sigma, self.min_alpha_sigma)

    def process(self, transitions, step):
        result = {}
        # Process per step
        self.memory.store(transitions)
        delta_t = step - self.time_t
        self.time_t = step
        self.learn_stamp += delta_t

        # Process per n_step
        if self.learn_stamp >= self.n_step:
            result = self.learn()
            if self.lr_decay:
                self.learning_rate_decay(step)
            self.learn_stamp = 0

        return result
