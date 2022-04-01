import torch

torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import os
import numpy as np

from .ppo import PPO
from core.network import Network
from core.optimizer import Optimizer


class ICM_PPO(PPO):
    """Intrinsic Curiosity Module (ICM) with PPO agent.

    Args:
        state_size (int): dimension of state.
        action_size (int): dimension of action.
        hidden_size (int): dimension of hidden unit.
        optim_config (dict): dictionary of the optimizer info.
            (key: 'name', value: name of optimizer)
        icm_network (str): key of network class in _network_dict.txt.
        beta (float): weight of the inverse model loss against the forward model loss.
        lamb (float): weight of the policy gradient loss against the intrinsic reward signal.
        eta (float): intrinsic reward scaling factor.
        extrinsic_coeff (float): coefficient of extrinsic reward.
        intrinsic_coeff (float): coefficient of intrinsic reward.
        obs_normalize (bool): parameter that determine whether to normalize observation.
        ri_normalize (bool): parameter that determine whether to normalize intrinsic reward.
        batch_norm (bool): parameter that determine whether to use batch normalization.
    """

    def __init__(
        self,
        state_size,
        action_size,
        hidden_size=512,
        optim_config={"name": "adam"},
        # Parameters for Curiosity-driven Exploration
        icm_network="icm_mlp",
        beta=0.2,
        lamb=1.0,
        eta=0.01,
        extrinsic_coeff=1.0,
        intrinsic_coeff=1.0,
        obs_normalize=True,
        ri_normalize=True,
        batch_norm=True,
        **kwargs,
    ):
        super(ICM_PPO, self).__init__(
            state_size=state_size,
            action_size=action_size,
            hidden_size=hidden_size,
            optim_config=optim_config,
            **kwargs,
        )
        self.obs_normalize = obs_normalize
        self.ri_normalize = ri_normalize
        self.batch_norm = batch_norm

        self.icm = Network(
            icm_network,
            state_size,
            action_size,
            self.num_workers,
            self.gamma,
            eta,
            self.action_type,
            ri_normalize,
            obs_normalize,
            batch_norm,
            D_hidden=hidden_size,
        ).to(self.device)
        self.icm_optimizer = Optimizer(**optim_config, params=self.icm.parameters())

        self.beta = beta
        self.lamb = lamb
        self.eta = eta
        self.extrinsic_coeff = extrinsic_coeff
        self.intrinsic_coeff = intrinsic_coeff

    def learn(self):
        transitions = self.memory.sample()
        for key in transitions.keys():
            transitions[key] = self.as_tensor(transitions[key])

        state = transitions["state"]
        action = transitions["action"]
        reward = transitions["reward"]
        next_state = transitions["next_state"]
        done = transitions["done"]

        # set prob_a_old and advantage
        with torch.no_grad():
            # ICM
            self.icm.update_rms_obs(next_state)
            r_i, _, _ = self.icm(state, action, next_state, update_ri=True)
            reward = (
                self.extrinsic_coeff * reward + self.intrinsic_coeff * r_i.unsqueeze(1)
            )

            if self.action_type == "continuous":
                mu, std, value = self.network(state)
                m = Normal(mu, std)
                z = torch.atanh(torch.clamp(action, -1 + 1e-7, 1 - 1e-7))
                log_prob = m.log_prob(z)
            else:
                pi, value = self.network(state)
                log_prob = pi.gather(1, action.long()).log()
            log_prob_old = log_prob

            next_value = self.network(next_state)[-1]
            delta = reward + (1 - done) * self.gamma * next_value - value
            adv = delta.clone()
            adv, done = adv.view(-1, self.n_step), done.view(-1, self.n_step)
            for t in reversed(range(self.n_step - 1)):
                adv[:, t] += (
                    (1 - done[:, t]) * self.gamma * self._lambda * adv[:, t + 1]
                )

            ret = adv.view(-1, 1) + value

            if self.use_standardization:
                adv = (adv - adv.mean(dim=1, keepdim=True)) / (
                    adv.std(dim=1, keepdim=True) + 1e-7
                )
            adv = adv.view(-1, 1)

        mean_ret = ret.mean().item()

        # start train iteration
        actor_losses, critic_losses, entropy_losses, ratios, probs = [], [], [], [], []
        idxs = np.arange(len(reward))
        for _ in range(self.n_epoch):
            np.random.shuffle(idxs)
            for offset in range(0, len(reward), self.batch_size):
                idx = idxs[offset : offset + self.batch_size]

                _state, _action, _value, _ret, _next_state, _adv, _log_prob_old = map(
                    lambda x: [_x[idx] for _x in x] if isinstance(x, list) else x[idx],
                    [state, action, value, ret, next_state, adv, log_prob_old],
                )

                if self.action_type == "continuous":
                    mu, std, value_pred = self.network(_state)
                    m = Normal(mu, std)
                    z = torch.atanh(torch.clamp(_action, -1 + 1e-7, 1 - 1e-7))
                    log_prob = m.log_prob(z)
                else:
                    pi, value_pred = self.network(_state)
                    m = Categorical(pi)
                    log_prob = m.log_prob(_action.squeeze(-1)).unsqueeze(-1)

                ratio = (log_prob - _log_prob_old).sum(1, keepdim=True).exp()
                surr1 = ratio * _adv
                surr2 = (
                    torch.clamp(
                        ratio, min=1 - self.epsilon_clip, max=1 + self.epsilon_clip
                    )
                    * _adv
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = _value + torch.clamp(
                    value_pred - _value, -self.epsilon_clip, self.epsilon_clip
                )

                critic_loss1 = F.mse_loss(value_pred, _ret)
                critic_loss2 = F.mse_loss(value_pred_clipped, _ret)

                critic_loss = torch.max(critic_loss1, critic_loss2).mean()
                entropy_loss = -m.entropy().mean()

                # ICM
                _, l_f, l_i = self.icm(_state, _action, _next_state)

                loss = self.lamb * (
                    actor_loss
                    + self.vf_coef * critic_loss
                    + self.ent_coef * entropy_loss
                )

                icm_loss = self.beta * l_f + (1 - self.beta) * l_i

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.clip_grad_norm
                )
                self.optimizer.step()

                self.icm_optimizer.zero_grad(set_to_none=True)
                icm_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.icm.parameters(), self.clip_grad_norm
                )
                self.icm_optimizer.step()

                probs.append(log_prob.exp().min().item())
                ratios.append(ratio.max().item())
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy_loss.item())

        result = {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "entropy_loss": np.mean(entropy_losses),
            "max_ratio": max(ratios),
            "min_prob": min(probs),
            "mean_ret": mean_ret,
            "r_i": r_i.mean().item(),
            "l_f": l_f.item(),
            "l_i": l_i.item(),
        }
        return result

    def save(self, path):
        print(f"...Save model to {path}...")
        torch.save(
            {
                "network": self.network.state_dict(),
                "icm": self.icm.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "icm_optimizer": self.icm_optimizer.state_dict(),
            },
            os.path.join(path, "ckpt"),
        )

    def load(self, path):
        print(f"...Load model from {path}...")
        checkpoint = torch.load(os.path.join(path, "ckpt"), map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.icm.load_state_dict(checkpoint["icm"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.icm_optimizer.load_state_dict(checkpoint["icm_optimizer"])
