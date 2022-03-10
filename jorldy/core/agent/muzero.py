from collections import deque
from itertools import islice
import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
import numpy as np

from .base import BaseAgent

class MuZero(BaseAgent):
    """MuZero agent.

    Args:
        -
    """

    def __init__(
        self,
        # MuZero
        **kwargs
    ):
        super(MuZero, self).__init__(network=network, **kwargs)
        

    @torch.no_grad()
    def act(self, state, training=True):
        self.network.train(training)
        pass

    def learn(self):
        pass
        self.num_learn += 1

        result = {
            "loss": loss.item(),
        }

        return result

    def process(self, transitions, step):
        pass
    
        return result


    def interact_callback(self, transition):
        pass

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
            
            reward_list.append(self.tree[node_id]['r']
            
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
                               
   def vec2scalar(self, vec)