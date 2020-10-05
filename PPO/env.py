import gym
from gym import spaces
import numpy as np
from scipy import ndimage

import os
from PIL import Image
import math

from collections import defaultdict


class Support_v0(gym.Env):
    def __init__(self):
        super().__init__()

        self.dataset_dir = "../size50/train/"
        _, _, self.filenames = next(os.walk(self.dataset_dir))

        self.board_size = 50

        # feature shape
        # 1) model
        # 2) support
        # 3) empty positions lower than the action row
        # 4) the upper row
        # 5) the action row
        # 6) legal action
        self.obs_shape = (self.board_size, self.board_size, 6)

        self.action_space = spaces.Discrete(self.board_size)
        self.observation_space = spaces.Box(0.0, 1.0, self.obs_shape, dtype=np.float32)

        self.reward = 1.0

    def is_valid_action(self, action):
        if self.model[self.action_row, action] or self.support[self.action_row, action]:
            return False
        else:
            return True

    def step(self, action):
        if not self.is_valid_action(action):
            return self.obs(), -9999.0, False, {}

        self.support[self.action_row, action] = True

        if self.update_action_row():
            if self.action_row == self.board_size:
                return np.ones(self.obs_shape, dtype=np.float32), -self.support_len(), True, {}
            else:
                return self.obs(), self.reward, False, {}
        else:
            return self.obs(), -self.reward / 100, False, {}

    def reset(self):
        filename = np.random.choice(self.filenames, 1)
        img = Image.open(self.dataset_dir + filename[0]).convert("L")
        self.model = np.array(img, dtype=np.bool)
        self.support = np.zeros_like(self.model)

        self.action_row = 1
        self.update_action_row()

        return self.obs()

    def _is_stable(self, row):
        upper = np.logical_or(self.model[row, :], self.support[row, :])
        lower = np.logical_or(self.model[row + 1, :], self.support[row + 1, :])

        dilated = ndimage.binary_dilation(lower)

        res = np.logical_and(upper, np.logical_not(dilated))
        supported = not res.any()
        return supported

    def update_action_row(self):
        action_row_change = False
        while self.action_row < self.board_size:
            if self._is_stable(self.action_row - 1):
                self.action_row += 1
                action_row_change = True
            else:
                break

        return action_row_change

    def empty_position_feature(self):
        filled = np.logical_or(self.model, self.support)

        # empty feature
        feature = np.logical_not(filled)

        # no need to consider rows upper than the action row
        feature[0 : self.action_row, :] = False
        return feature

    def row_feature(self, row):
        filled = np.logical_or(self.model[row, :], self.support[row, :])

        # tile the row along the axis 0
        feature = np.tile(filled, (self.board_size, 1))
        return feature

    def upper_row_feature(self):
        return self.row_feature(self.action_row - 1)

    def action_row_feature(self):
        return self.row_feature(self.action_row)

    def legal_action_feature(self):
        filled = np.logical_or(
            self.model[self.action_row, :], self.support[self.action_row, :]
        )
        empty = np.logical_not(filled)
        feature = np.zeros_like(self.model)
        feature[self.action_row, np.nonzero(empty)[0]] = True
        return feature

    def obs(self):
        return np.stack(
            [
                self.model,
                self.support,
                self.empty_position_feature(),
                self.upper_row_feature(),
                self.action_row_feature(),
                self.legal_action_feature(),
            ],
            axis=2,
        ).astype(np.float32)

    def support_len(self):
        def get_neighbors(pos):
            # return 8-connected neighbors
            neighbors = []
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    # continue self
                    if i == 0 and j == 0:
                        continue

                    # row, col
                    ni, nj = pos[0] + i, pos[1] + j

                    # check if outside
                    if ni < 0 or ni >= self.board_size:
                        continue
                    if nj < 0 or nj >= self.board_size:
                        continue

                    neighbors.append((ni, nj))
            return neighbors

        def distance(pos1, pos2):
            dx = abs(pos2[0] - pos1[0])
            dy = abs(pos2[1] - pos1[1])
            if dx + dy > 1:
                return math.sqrt(2)
            else:
                return 1

        last_label = 0
        stack = []
        label = {}
        length = defaultdict(float)

        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.support[i, j]:
                    if (i, j) in label.keys():
                        continue
                    else:
                        last_label += 1
                        label[(i, j)] = last_label
                        stack.append((i, j))

                        while stack:
                            top = stack.pop()
                            neighbors = get_neighbors(top)
                            for neighbor in neighbors:
                                if (
                                    self.support[neighbor]
                                    and not neighbor in label.keys()
                                ):
                                    stack.append(neighbor)
                                    label[neighbor] = last_label
                                    length[last_label] += distance(top, neighbor)
        
        return sum(length.values())


if __name__ == "__main__":

    env = Support_v0()
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)