from abc import *


class BaseEnv(ABC):
    @abstractmethod
    def reset(self):
        """
        Reset env and return initial state.

        Parameter Type / Shape
        - state: ndarray / (N_batch, D_state) ex) (1, 4), (1, 4, 84, 84)
        """
        state = None
        return state

    @abstractmethod
    def step(self, action):
        """
        Through action, one step proceeds according to the dynamics of the environment.

        Parameter Type / Shape
        - action:   ndarray / (N_batch, *D_action) ex) (1, 3), (1, 1)
        - state:    ndarray / (N_batch, D_state) ex) (1, 4), (1, 4, 84, 84)
        - reward:   ndarray / (N_batch, D_reward) ex) (1, 1)
        - done:     ndarray / (N_batch, D_done) ex) (1, 1)
        """
        state, reward, done = None, None, None
        return state, reward, done

    @abstractmethod
    def close(self):
        """
        Close the environment
        """
        pass

    def recordable(self):
        return False
