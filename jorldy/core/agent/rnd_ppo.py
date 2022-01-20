import torch

torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import os
import numpy as np

from .ppo import PPO
from core.network import Network


class RND_PPO(PPO):
    """Random Network Distillation (RND) with PPO agent.

    Args:
        state_size (int): dimension of state.
        action_size (int): dimension of action.
        hidden_size (int): dimension of hidden unit.
        rnd_network (str): key of network class in _network_dict.txt.
        gamma_i (float): discount factor of intrinsic reward.
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
        # Parameters for Random Network Distillation
        rnd_network="rnd_mlp",
        gamma_i=0.99,
        extrinsic_coeff=1.0,
        intrinsic_coeff=1.0,
        obs_normalize=True,
        ri_normalize=True,
        batch_norm=True,
        **kwargs,
    ):
        super(RND_PPO, self).__init__(
            state_size=state_size,
            action_size=action_size,
            hidden_size=hidden_size,
            **kwargs,
        )

        self.rnd_network = rnd_network

        self.gamma_i = gamma_i
        self.extrinsic_coeff = extrinsic_coeff
        self.intrinsic_coeff = intrinsic_coeff

        self.obs_normalize = obs_normalize
        self.ri_normalize = ri_normalize
        self.batch_norm = batch_norm

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

        self.optimizer.add_param_group({"params": self.rnd.parameters()})

        # Freeze random network
        for name, param in self.rnd.named_parameters():
            if "target" in name:
                param.requires_grad = False

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

        # set pi_old and advantage
        with torch.no_grad():
            # RND: calculate exploration reward, update moments of obs and r_i
            self.rnd.update_rms_obs(next_state)
            r_i = self.rnd(next_state, update_ri=True)
            r_i = r_i.unsqueeze(-1)

            # Scaling extrinsic and intrinsic reward
            reward *= self.extrinsic_coeff
            r_i *= self.intrinsic_coeff

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
            v_i_old = v_i

            next_value = self.network(next_state)[-1]
            next_v_i = self.network.get_v_i(next_state)
            delta = reward + (1 - done) * self.gamma * next_value - value
            # non-episodic intrinsic reward, hence (1-done) not applied
            delta_i = r_i + self.gamma_i * next_v_i - v_i
            adv, adv_i = delta.clone(), delta_i.clone()
            adv, adv_i, done = (
                adv.view(-1, self.n_step),
                adv_i.view(-1, self.n_step),
                done.view(-1, self.n_step),
            )
            for t in reversed(range(self.n_step - 1)):
                adv[:, t] += (
                    (1 - done[:, t]) * self.gamma * self._lambda * adv[:, t + 1]
                )
                adv_i[:, t] += self.gamma_i * self._lambda * adv_i[:, t + 1]

            if self.use_standardization:
                adv = (adv - adv.mean(dim=1, keepdim=True)) / (
                    adv.std(dim=1, keepdim=True) + 1e-7
                )
                adv_i = (adv_i - adv_i.mean(dim=1, keepdim=True)) / (
                    adv_i.std(dim=1, keepdim=True) + 1e-7
                )
            adv = adv.view(-1, 1)
            adv_i = adv_i.view(-1, 1)
            done = done.view(-1, 1)

            ret = adv + value
            ret_i = adv_i + v_i

        mean_adv = adv.mean().item()
        mean_adv_i = adv_i.mean().item()
        mean_ret = ret.mean().item()
        mean_ret_i = ret_i.mean().item()

        # start train iteration
        actor_losses, critic_losses, entropy_losses, rnd_losses, ratios, probs = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        idxs = np.arange(len(reward))
        for idx_epoch in range(self.n_epoch):
            np.random.shuffle(idxs)
            for offset in range(0, len(reward), self.batch_size):
                idx = idxs[offset : offset + self.batch_size]

                (
                    _state,
                    _action,
                    _value,
                    _v_i_old,
                    _ret,
                    _ret_i,
                    _next_state,
                    _adv,
                    _adv_i,
                    _log_prob_old,
                ) = map(
                    lambda x: [_x[idx] for _x in x] if isinstance(x, list) else x[idx],
                    [
                        state,
                        action,
                        v_i_old,
                        value,
                        ret,
                        ret_i,
                        next_state,
                        adv,
                        adv_i,
                        log_prob_old,
                    ],
                )

                _r_i = self.rnd.forward(_next_state) * self.intrinsic_coeff

                if self.action_type == "continuous":
                    mu, std, value_pred = self.network(_state)
                    m = Normal(mu, std)
                    z = torch.atanh(torch.clamp(_action, -1 + 1e-7, 1 - 1e-7))
                    log_prob = m.log_prob(z)
                else:
                    pi, value_pred = self.network(_state)
                    m = Categorical(pi)
                    log_prob = m.log_prob(_action.squeeze(-1)).unsqueeze(-1)
                v_i = self.network.get_v_i(_state)

                ratio = (log_prob - _log_prob_old).sum(1, keepdim=True).exp()
                surr1 = ratio * (_adv + _adv_i)
                surr2 = torch.clamp(
                    ratio, min=1 - self.epsilon_clip, max=1 + self.epsilon_clip
                ) * (_adv + _adv_i)
                actor_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = _value + torch.clamp(
                    value_pred - _value, -self.epsilon_clip, self.epsilon_clip
                )

                critic_loss1 = F.mse_loss(value_pred, _ret)
                critic_loss2 = F.mse_loss(value_pred_clipped, _ret)

                v_i_clipped = _v_i_old + torch.clamp(
                    v_i - _v_i_old, -self.epsilon_clip, self.epsilon_clip
                )

                critic_i_loss1 = F.mse_loss(v_i, _ret_i)
                critic_i_loss2 = F.mse_loss(v_i_clipped, _ret_i)

                critic_loss = (
                    torch.max(critic_loss1, critic_loss2).mean()
                    + torch.max(critic_i_loss1, critic_i_loss2).mean()
                )

                entropy_loss = -m.entropy().mean()
                ppo_loss = (
                    actor_loss
                    + self.vf_coef * critic_loss
                    + self.ent_coef * entropy_loss
                )
                rnd_loss = _r_i.mean()

                loss = ppo_loss + rnd_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.clip_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.rnd.parameters(), self.clip_grad_norm
                )
                self.optimizer.step()

                probs.append(log_prob.exp().min().item())
                ratios.append(ratio.max().item())
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy_loss.item())
                rnd_losses.append(rnd_loss.item())

        result = {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "entropy_loss": np.mean(entropy_losses),
            "r_i": np.mean(rnd_losses),
            "max_ratio": max(ratios),
            "min_prob": min(probs),
            "mean_adv": mean_adv,
            "mean_adv_i": mean_adv_i,
            "mean_ret": mean_ret,
            "mean_ret_i": mean_ret_i,
        }
        return result

    def process(self, transitions, step):
        result = {}
        # Process per step
        self.memory.store(transitions)
        delta_t = step - self.time_t
        self.time_t = step
        self.learn_stamp += delta_t

        if len(transitions) > 0 and transitions[0]["done"]:
            self.state_seq = None

        # Process per epi
        if self.learn_stamp >= self.n_step:
            result = self.learn()
            self.learn_stamp = 0

        return result

    def save(self, path):
        print(f"...Save model to {path}...")
        torch.save(
            {
                "network": self.network.state_dict(),
                "rnd": self.rnd.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            os.path.join(path, "ckpt"),
        )

    def load(self, path):
        print(f"...Load model from {path}...")
        checkpoint = torch.load(os.path.join(path, "ckpt"), map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.rnd.load_state_dict(checkpoint["rnd"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
