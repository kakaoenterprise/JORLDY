import numpy as np
from .base import BaseEnv


class Tictactoe(BaseEnv):
    """TicTacToe environment.

    Args:
        env_name (str): name of environment in BoardGame.
        render (bool): parameter that determine whether to render.
        input_type (str): parameter that determine type of inputs (image, vector).
        img_width (int): width of image input.
        img_height (int): height of image input.
        opponent_policy (str): Policy of the opponent (random)
    """

    def __init__(
        self,
        render=False,
        input_type="image",
        img_width=40,
        img_height=40,
        opponent_policy="random",
        **kwargs
    ):
        self.render = render
        self.input_type = input_type
        self.img_width = img_width
        self.img_height = img_height
        self.opponent_policy = opponent_policy

        self.score = 0

        self.state_size = (
            [1, img_height, img_width] if self.input_type == "image" else 9
        )
        self.action_size = 9
        self.action_type = "discrete"

        # None: 0 / O: 1 / X: -1
        self.gameboard = np.zeros([3, 3])

    def reset(self):
        self.score = 0
        self.gameboard = np.zeros([3, 3])

        state = self.state_processing(self.gameboard)
        return state

    def step(self, action):
        row = action // 3
        column = action % 3

        # Agent action
        if self.gameboard[row, column] == 0:
            self.gameboard[row, column] = 1
            reward, done = self.check_win(self.gameboard)

            # Opponent action
            if done == False:
                if self.opponent_policy == "random":
                    legal_idx = np.argwhere(self.gameboard == 0)

                    if len(legal_idx) > 0:
                        rand_idx = np.random.randint(legal_idx.shape[0])

                        row = legal_idx[rand_idx][0]
                        column = legal_idx[rand_idx][1]

                        self.gameboard[row, column] = -1

                reward, done = self.check_win(self.gameboard)
        else:
            reward, done = self.check_win(self.gameboard)

            if done == False:
                reward = np.array([-0.1])
                done = True

        next_state = self.state_processing(self.gameboard)
        self.score += reward[0]

        reward, done = map(lambda x: np.expand_dims(x, 0), [reward, [done]])
        return (next_state, reward, done)

    def state_processing(self, gameboard):
        if self.input_type == "image":
            gameboard_img = np.zeros([self.img_height, self.img_width])
            gameboard_img[:3, :3] = gameboard
            state = np.expand_dims(gameboard_img, axis=(0, 1)) * 255
        else:
            state = np.reshape(gameboard, (1, -1))
        return state

    def close(self):
        pass

    def check_win(self, gameboard):
        reward = np.array([0])
        done = False

        legal_idx = np.argwhere(gameboard == 0)

        sum_row = np.sum(gameboard, axis=0)
        sum_col = np.sum(gameboard, axis=1)
        sum_diag1 = np.trace(gameboard)
        sum_diag2 = np.trace(np.rot90(gameboard))

        if 3 in sum_row or 3 in sum_col or sum_diag1 == 3 or sum_diag2 == 3:
            reward = np.array([1])
            done = True
        elif -3 in sum_row or -3 in sum_col or sum_diag1 == -3 or sum_diag2 == -3:
            reward = np.array([-1])
            done = True

        if len(legal_idx) == 0 and done == False:
            reward = np.array([0.1])
            done = True

        return (reward, done)
