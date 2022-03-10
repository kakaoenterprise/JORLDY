from collections import deque
import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
import numpy as np

from .base import BaseAgent
from core.buffer import PERBuffer

class MuZero(BaseAgent):
    """MuZero agent.

    Args:
        -
    """

    def __init__(
        self,
        # MuZero
        network="pseudo",
        state_size=(96,96,3),
        action_size=88,
        batch_size=16,
        start_train_step=0,
        num_stacked_observation=32,
        buffer_size=100000,
        run_step=1e6,
        # PER
        alpha=0.6,
        beta=0.4,
        learn_period=4,
        uniform_sample_prob=1e-3,
        **kwargs
    ):
        super(MuZero, self).__init__(network=network, **kwargs)
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_state_shape = (6, 6, 1024)
        self.batch_size = batch_size
        self.start_train_step = start_train_step

        self.num_stacked_observation = num_stacked_observation
        self.stacked_observation = deque(maxlen=self.num_stacked_observation*2+1)

        self.num_learn

        # PER
        self.alpha = alpha
        self.beta = beta
        self.learn_period = learn_period
        self.learn_period_stamp = 0
        self.uniform_sample_prob = uniform_sample_prob
        self.beta_add = (1 - beta) / run_step
        self.buffer_size = buffer_size
        self.memory = PERBuffer(self.buffer_size, uniform_sample_prob)

        self.num_learn = 0

    def reset_observation(self):
        self.stacked_observation.clear()
        self.stacked_observation.extend([np.ones(self.state_size)])

    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training)
        self.stacked_observation.append(state)

        root_state = self.pseudo_representation(self.stacked_observation)
        mcts = MCTS()
        action, pi,  = mcts.run_mcts(root_state)

        self.stacked_observation.append(action)
        return action

    def learn(self):
        pass
        self.num_learn += 1

        result = {
            "loss": loss.item(),
        }

        return result

    def process(self, transitions, step):
        result = {}

        # Process per step
        delta_t = step - self.time_t
        self.memory.store(transitions)
        self.time_t = step
        self.learn_period_stamp += delta_t

        if (
            self.learn_period_stamp >= self.learn_period
            and self.memory.buffer_counter >= self.batch_size
            and self.time_t >= self.start_train_step
        ):
            result = self.learn()
            self.learning_rate_decay(step)
            self.learn_period_stamp = 0
    
        return result

    def interact_callback(self, transition):
        

        return _transition


    def save(self, path):
        print(f"...Save model to {path}...")
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            os.path.join(path, "ckpt"),
        )

    def load(self, path):
        print(f"...Load model from {path}...")
        checkpoint = torch.load(os.path.join(path, "ckpt"), map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.target_network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def pseudo_representation(self, stacked_observation):
        hidden_state_0 = torch.zeros(self.hidden_state_shape)
        return hidden_state_0

    def pseudo_prediction(self, hidden_state):
        pi = torch.zeros(self.action_size)
        value = 0
        return pi, value

    def pseudo_dynamics(self, hidden_state, action):
        next_hidden_state, reward = torch.zeros(self.hidden_state_shape), 0
        return next_hidden_state, reward
        

class MCTS():
    def __init__(self, p_fn, d_fn, action_size, n_mcts, n_unroll, gamma):
        self.p_fn = p_fn # prediction function
        self.d_fn = d_fn # dynamics function
        
        self.action_size = action_size
        self.n_mcts = n_mcts
        self.n_unroll = n_unroll+1

        self.gamma = gamma
        self.temp_param = 1.0
            
        self.c1 = 1.25
        self.c2 = 19652
        self.alpha = 0.3

        self.q_min = 0
        self.q_max = 0

        self.root_id = (0,)
        self.tree = {}

    
    def run_mcts(self, root_state):
        self.tree = self.init_mcts(root_state)
        
        for i in range(self.n_mcts):
            # selection
            leaf_id, leaf_state = self.selection(root_state)

            # expansion and evaluation
            leaf_v = self.expansion(leaf_id, leaf_state)

            # backup
            self.backup(leaf_id, leaf_v)

        root_value = self.tree[self.root_id]['q']
        root_action, pi = self.select_root_action()

        return root_action, pi, root_value
    
    def selection(self, root_state):
        node_id = self.root_id
        node_state = root_state 
        
        while self.tree[node_id]['n'] > 0:
            if len(node_id) <= self.n_unroll:
                UCB_list = []
                total_n = self.tree[node_id]['n']

                # for action_idx in self.tree[node_id]['child']:
                #     edge_id = node_id + (action_idx,)
                #     n = self.tree[edge_id]['n']
                #     total_n += n
                                
                for action_index in self.tree[node_id]['child']:
                    child_id = node_id + (action_index,)
                    n = self.tree[child_id]['n']
                    q = (self.tree[child_id]['q'] - self.q_min) / (self.q_max - self.q_min)
                    p = self.tree[node_id]['p'][0,action_index]
                    u = (p * np.sqrt(total_n) / (n + 1)) * (self.c1 + np.log((total_n+self.c2+1)/self.c2))
                    UCB_list.append(q + u)
                
                a_UCB = np.argmax(UCB_list) 
                node_id += (a_UCB,)
                node_state, _ = self.d_fn(node_state, a_UCB) # a_UCB를 network의 입력형태로 변환 필요 
            else:
                break
                
        return node_id, node_state
    
    def expansion(self, leaf_id, leaf_state):
        for action_idx in range(self.action_size):
            child_id = leaf_id + (action_idx,)
            
            s_child, r_child = self.d_fn(leaf_state, action_idx) # action_idx를 network의 입력형태로 변환 필요 
            # r_child를 scalar 형태로 변환 -> 네트워크에서 구현? 
            
            p_child, _ = self.p_fn(s_child)
            
            self.tree[child_id] = {'child': [],
                                   'n': 0.,
                                   'q': 0.,
                                   'p': p_child,
                                   'r': r_child_scalar}

            self.tree[leaf_id]['child'].append(action_idx)    
        
        _, leaf_v = self.p_fn(leaf_state)
        # v를 scalar 형태로 변환 -> 네트워크에서 구현? 

        return leaf_v 
        
    def backup(self, leaf_id, leaf_v):
        node_id = leaf_id
        node_v = leaf_v
        reward_list = [self.tree[node_id]['r']]
        
        while True:
            # Calculate G
            discount_sum_r = 0
            n = len(reward_list)-1

            for i in range(len(reward_list)):
                discount_sum_r += (self.gamma**(n-i)) * reward_list[i]

            G = discount_sum_r + ((self.gamma**(n+1))*value)
            
            # Update Q and N
            q = (self.tree[node_id]['n']*self.tree[node_id]['q'] + G) / (self.tree[node_id]['n']+1)
            self.tree[node_id]['q'] = q
            self.tree[node_id]['n'] += 1
            
            # Update max q and min q
            self.q_max = max(q, self.q_max)
            self.q_min = min(q, self.q_min)
            
            node_id = node_id[:-1]
            
            if node_id == ():
                break
            
            reward_list.append(self.tree[node_id]['r'])
            
    def init_mcts(self, root_state):
        tree = {}
        root_id = (0,)

        p_root, _ = self.p_fn(root_state)

        # init root node
        tree[root_id] = {'child': [],
                         'n': 0.,
                         'q': 0.,
                         'p': p_root,
                         'r': 0.}
        
        return tree 
        
    def select_root_action(self):
        child = self.tree[self.root_id]['child']

        n_list = []

        for child_num in child:
            child_idx = self.root_id + (child_num,)
            n_list.append(self.tree[child_idx]['n'])

        pi = np.asarray(n_list) / np.sum(n_list)
        pi_temp = (np.asarray(n_list) ** (1/self.temp_param)) / (np.sum(n_list) ** (1/self.temp_param))
                               
        noise_probs = self.alpha * np.random.dirichlet(np.ones(self.action_size))
        pi_noise = pi_temp + noise_probs
        pi_noise = pi_noise / np.sum(pi_noise)

        action_idx = np.random.choice(self.action_size, p=pi_noise)

        return action_idx, pi

    def backup(self):
        pass


class History():
    def __init__(self):
        pass

    def append(self, transition):
        pass

