import torch

torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import os
import numpy as np

from .ppo import PPO
from core.network import Network
from core.optimizer import Optimizer


class RND_PPO(PPO):
    """Random Network Distillation (RND) with PPO agent.

    Args:
        state_size (int): dimension of state.
        action_size (int): dimension of action.
        hidden_size (int): dimension of hidden unit.
        optim_config (dict): dictionary of the optimizer info.
            (key: 'name', value: name of optimizer)
        rnd_network (str): key of network class in _network_dict.txt.
        gamma_i (float): discount factor of intrinsic reward.
        extrinsic_coeff (float): coefficient of extrinsic reward.
        intrinsic_coeff (float): coefficient of intrinsic reward.
        obs_normalize (bool): parameter that determine whether to normalize observation.
        ri_normalize (bool): parameter that determine whether to normalize intrinsic reward.
        batch_norm (bool): parameter that determine whether to use batch normalization.
        non_episodic (bool): parameter that determine whether to use non episodic return(only intrinsic).
        non_extrinsic (bool): parameter that determine whether to use intrinsic reward only.
    """

    def __init__(
        self,
        state_size,
        action_size,
        hidden_size=512,
        optim_config={"name": "adam"},
        # Parameters for Random Network Distillation
        rnd_network="rnd_mlp",
        gamma_i=0.99,
        extrinsic_coeff=2.0,
        intrinsic_coeff=1.0,
        obs_normalize=True,
        ri_normalize=True,
        batch_norm=True,
        non_episodic=True,
        non_extrinsic=False,
        **kwargs,
    ):
        super(RND_PPO, self).__init__(
            state_size=state_size,
            action_size=action_size,
            hidden_size=hidden_size,
            optim_config=optim_config,
            **kwargs,
        )

        self.rnd_network = rnd_network

        self.gamma_i = gamma_i
        self.extrinsic_coeff = extrinsic_coeff
        self.intrinsic_coeff = intrinsic_coeff

        self.obs_normalize = obs_normalize
        self.ri_normalize = ri_normalize
        self.batch_norm = batch_norm
        self.non_episodic = non_episodic
        self.non_extrinsic = non_extrinsic

        self.rnd = Network(
            rnd_network,
            state_size,
            action_size,
            self.num_workers,
            gamma_i,
            ri_normalize,
            obs_normalize,
            batch_norm,
            D_hidden=hidden_size,
        ).to(self.device)
        self.rnd_optimizer = Optimizer(**optim_config, params=self.rnd.parameters())

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

        # use extrinsic check
        if self.non_extrinsic:
            reward *= 0.0

        # set pi_old and advantage
        with torch.no_grad():
            # RND: calculate exploration reward, update moments of obs and r_i
            self.rnd.update_rms_obs(next_state)
            r_i = self.rnd(next_state, update_ri=True)

            if self.action_type == "continuous":
                mu, std, value = self.network(state)
                m = Normal(mu, std)
                z = torch.atanh(torch.clamp(action, -1 + 1e-7, 1 - 1e-7))
                log_prob = m.log_prob(z)
            else:
                pi, value = self.network(state)
                log_prob = pi.gather(1, action.long()).log()
            log_prob_old = log_prob
            v_i = self.network.get_v_i(state)

            next_value = self.network(next_state)[-1]
            delta = reward + (1 - done) * self.gamma * next_value - value

            next_v_i = self.network.get_v_i(next_state)
            episodic_factor = 1.0 if self.non_episodic else (1 - done)
            delta_i = r_i + episodic_factor * self.gamma_i * next_v_i - v_i

            adv, adv_i = delta.clone(), delta_i.clone()
            adv, adv_i = adv.view(-1, self.n_step), adv_i.view(-1, self.n_step)
            done = done.view(-1, self.n_step)

            for t in reversed(range(self.n_step - 1)):
                adv[:, t] += (
                    (1 - done[:, t]) * self.gamma * self._lambda * adv[:, t + 1]
                )
                episodic_factor = 1.0 if self.non_episodic else (1 - done[:, t])
                adv_i[:, t] += (
                    episodic_factor * self.gamma_i * self._lambda * adv_i[:, t + 1]
                )

            ret = adv.view(-1, 1) + value
            ret_i = adv_i.view(-1, 1) + v_i

            adv = self.extrinsic_coeff * adv + self.intrinsic_coeff * adv_i

            if self.use_standardization:
                adv = (adv - adv.mean(dim=1, keepdim=True)) / (
                    adv.std(dim=1, keepdim=True) + 1e-7
                )

            adv, done = adv.view(-1, 1), done.view(-1, 1)

        mean_ret = ret.mean().item()
        mean_ret_i = ret_i.mean().item()

        # start train iteration
        actor_losses, critic_e_losses, critic_i_losses = [], [], []
        entropy_losses, rnd_losses, ratios, probs = [], [], [], []
        idxs = np.arange(len(reward))
        for idx_epoch in range(self.n_epoch):
            np.random.shuffle(idxs)
            for offset in range(0, len(reward), self.batch_size):
                idx = idxs[offset : offset + self.batch_size]
                (
                    _state,
                    _action,
                    _value,
                    _v_i,
                    _ret,
                    _ret_i,
                    _next_state,
                    _adv,
                    _log_prob_old,
                ) = map(
                    lambda x: [_x[idx] for _x in x] if isinstance(x, list) else x[idx],
                    [
                        state,
                        action,
                        value,
                        v_i,
                        ret,
                        ret_i,
                        next_state,
                        adv,
                        log_prob_old,
                    ],
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
                value_i = self.network.get_v_i(_state)

                ratio = (log_prob - _log_prob_old).sum(1, keepdim=True).exp()
                surr1 = ratio * _adv
                surr2 = (
                    torch.clamp(
                        ratio, min=1 - self.epsilon_clip, max=1 + self.epsilon_clip
                    )
                    * _adv
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic Clipping
                value_pred_clipped = _value + torch.clamp(
                    value_pred - _value, -self.epsilon_clip, self.epsilon_clip
                )

                critic_loss1 = F.mse_loss(value_pred, _ret)
                critic_loss2 = F.mse_loss(value_pred_clipped, _ret)

                critic_e_loss = torch.max(critic_loss1, critic_loss2).mean()

                # Critic Clipping (intrinsic)
                value_i_clipped = _v_i + torch.clamp(
                    value_i - _v_i, -self.epsilon_clip, self.epsilon_clip
                )

                critic_i_loss1 = F.mse_loss(value_i, _ret_i)
                critic_i_loss2 = F.mse_loss(value_i_clipped, _ret_i)

                critic_i_loss = torch.max(critic_i_loss1, critic_i_loss2).mean()

                critic_loss = critic_e_loss + critic_i_loss

                entropy_loss = -m.entropy().mean()
                loss = (
                    actor_loss
                    + self.vf_coef * critic_loss
                    + self.ent_coef * entropy_loss
                )

                _r_i = self.rnd.forward(_next_state)
                rnd_loss = _r_i.mean()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.clip_grad_norm
                )
                self.optimizer.step()

                self.rnd_optimizer.zero_grad(set_to_none=True)
                rnd_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.rnd.parameters(), self.clip_grad_norm
                )
                self.rnd_optimizer.step()

                probs.append(log_prob.exp().min().item())
                ratios.append(ratio.max().item())
                actor_losses.append(actor_loss.item())
                critic_e_losses.append(critic_e_loss.item())
                critic_i_losses.append(critic_i_loss.item())
                entropy_losses.append(entropy_loss.item())
                rnd_losses.append(rnd_loss.item())

        result = {
            "actor_loss": np.mean(actor_losses),
            "critic_e_loss": np.mean(critic_e_losses),
            "critic_i_loss": np.mean(critic_i_losses),
            "entropy_loss": np.mean(entropy_losses),
            "r_i": np.mean(rnd_losses),
            "max_ratio": max(ratios),
            "min_prob": min(probs),
            "mean_ret": mean_ret,
            "mean_ret_i": mean_ret_i,
        }
        return result

    def save(self, path):
        print(f"...Save model to {path}...")
        torch.save(
            {
                "network": self.network.state_dict(),
                "rnd": self.rnd.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "rnd_optimizer": self.rnd_optimizer.state_dict(),
            },
            os.path.join(path, "ckpt"),
        )

    def load(self, path):
        print(f"...Load model from {path}...")
        checkpoint = torch.load(os.path.join(path, "ckpt"), map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.rnd.load_state_dict(checkpoint["rnd"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.rnd_optimizer.load_state_dict(checkpoint["rnd_optimizer"])
