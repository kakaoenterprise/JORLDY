import torch
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np

from .reinforce import REINFORCE

class PPO(REINFORCE):
    """PPO agent. 
    
    Args: 
        batch_size (int): the number of samples in the one batch.
        n_step (int): The number of steps to run for each environment per update.
        n_epoch (int): Number of epoch when optimizing the surrogate.
        _lambda (float): Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        epsilon_clip (float): probability ratio clipping interval.
        vf_coef (float): Value function coefficient for the loss calculation.
        ent_coef (float): Entropy coefficient for the loss calculation.
        clip_grad_norm (float): gradient clipping threshold.
        num_workers: the number of agents in distributed learning.
    """
    def __init__(self,
                 batch_size=32,
                 n_step=128,
                 n_epoch=3,
                 _lambda=0.95,
                 epsilon_clip=0.1,
                 vf_coef=1.0,
                 ent_coef=0.01,
                 clip_grad_norm=1.0,
                 num_workers=1,
                 **kwargs,
                 ):
        super(PPO, self).__init__(**kwargs)
        
        self.batch_size = batch_size
        self.n_step = n_step
        self.n_epoch = n_epoch
        self._lambda = _lambda
        self.epsilon_clip = epsilon_clip
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.clip_grad_norm = clip_grad_norm
        self.num_workers = num_workers
        self.time_t = 0
        self.learn_stamp = 0
    
    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training)
        
        if self.action_type == "continuous":
            mu, std, _ = self.network(torch.as_tensor(state, dtype=torch.float32, device=self.device))
            z = torch.normal(mu, std) if training else mu
            action = torch.tanh(z)
        else:
            pi, _ = self.network(torch.as_tensor(state, dtype=torch.float32, device=self.device))
            action = torch.multinomial(pi, 1) if training else torch.argmax(pi, dim=-1, keepdim=True)
        return {'action': action.cpu().numpy()}

    def learn(self):
        transitions = self.memory.sample()
        for key in transitions.keys():
            transitions[key] = torch.as_tensor(transitions[key], dtype=torch.float32, device=self.device)

        state = transitions['state']
        action = transitions['action']
        reward = transitions['reward']
        next_state = transitions['next_state']
        done = transitions['done']
        
        # set prob_a_old and advantage
        with torch.no_grad():            
            if self.action_type == "continuous":
                mu, std, value = self.network(state)
                m = Normal(mu, std)
                z = torch.atanh(torch.clamp(action, -1+1e-7, 1-1e-7))
                prob = m.log_prob(z).exp()
            else:
                pi, value = self.network(state)
                prob = pi.gather(1, action.long())
            prob_old = prob
            
            next_value = self.network(next_state)[-1]
            delta = reward + (1 - done) * self.gamma * next_value - value
            adv = delta.clone()
            adv, done = adv.view(-1, self.n_step), done.view(-1, self.n_step)
            for t in reversed(range(self.n_step - 1)):
                adv[:, t] += (1 - done[:, t]) * self.gamma * self._lambda * adv[:, t+1]
            if self.use_standardization:
                adv = (adv - adv.mean(dim=1, keepdim=True)) / (adv.std(dim=1, keepdim=True) + 1e-7)
            adv = adv.view(-1, 1)
            ret = adv + value
        
        # start train iteration
        actor_losses, critic_losses, entropy_losses, ratios, probs = [], [], [], [], []
        idxs = np.arange(len(reward))
        for _ in range(self.n_epoch):
            np.random.shuffle(idxs)
            for offset in range(0, len(reward), self.batch_size):
                idx = idxs[offset : offset + self.batch_size]
                
                _state, _action, _ret, _next_state, _adv, _prob_old =\
                    map(lambda x: x[idx], [state, action, ret, next_state, adv, prob_old])

                if self.action_type == "continuous":
                    mu, std, value = self.network(_state)
                    m = Normal(mu, std)
                    z = torch.atanh(torch.clamp(_action, -1+1e-7, 1-1e-7))
                    prob = m.log_prob(z).exp()
                else:
                    pi, value = self.network(_state)
                    m = Categorical(pi)
                    prob = pi.gather(1, _action.long())
                
                ratio = (prob / (_prob_old + 1e-7)).prod(1, keepdim=True)
                surr1 = ratio * _adv
                surr2 = torch.clamp(ratio, min=1-self.epsilon_clip, max=1+self.epsilon_clip) * _adv
                actor_loss = -torch.min(surr1, surr2).mean() 
                
                critic_loss = F.mse_loss(value, _ret).mean()
                
                entropy_loss = -m.entropy().mean()
                
                loss = actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                
                probs.append(prob.min().item())
                ratios.append(ratio.max().item())
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy_loss.item())
        result = {
            'actor_loss' : np.mean(actor_losses),
            'critic_loss' : np.mean(critic_losses),
            'entropy_loss' : np.mean(entropy_losses),
            'max_ratio' : max(ratios),
            'min_prob': min(probs),
            'min_prob_old': prob_old.min().item(),
        }
        return result

    def process(self, transitions, step):
        result = {}
        # Process per step
        self.memory.store(transitions)
        delta_t = step - self.time_t
        self.time_t = step
        self.learn_stamp += delta_t
        
        # Process per epi
        if self.learn_stamp >= self.n_step :
            result = self.learn()
            self.learn_stamp = 0
        
        return result