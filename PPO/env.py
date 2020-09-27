import gym
from gym import spaces
import numpy as np
from scipy import ndimage

import cv2


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = spaces.Box(0, 255, new_shape, dtype=np.uint8)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super(ScaledFloatFrame, self).__init__(env)

        self.observation_space = spaces.Box(
            0.0, 1.0, self.observation_space.shape, dtype=np.float32
        )

    def observation(self, obs):
        return obs.astype(np.float32) / 255.0

class ROIWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ROIWrapper, self).__init__(env)

        old_shape = self.observation_space.shape
        new_shape = (old_shape[0], old_shape[1], old_shape[2]+1)
        self.observation_space = spaces.Box(
            0, 255, new_shape, dtype=np.uint8
        )

    def _ROI(self):
        img = np.zeros_like(self.model)
        img[self.action_row-1:,:] = 255
        return img

    def observation(self, obs):
        return np.concatenate((obs, self._ROI()[...,np.newaxis]), axis=2)


class LegalActionWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(LegalActionWrapper, self).__init__(env)

        old_shape = self.observation_space.shape
        new_shape = (old_shape[0], old_shape[1], old_shape[2]+1)
        self.observation_space = spaces.Box(
            0, 255, new_shape, dtype=np.uint8
        )

    def _legal_action_image(self):
        img = np.zeros_like(self.model)

        if self.action_row == self.height:
            img[0, :] = 255
            return img

        model_empty = self.model[self.action_row, :] == 0
        support_empty = self.support[self.action_row, :] == 0
        empty = np.logical_and(model_empty, support_empty)
        legal_actions = np.nonzero(empty)[0]

        img[self.action_row, legal_actions] = 255
        return img

    def observation(self, obs):
        return np.concatenate((obs, self._legal_action_image()[...,np.newaxis]), axis=2)


class SupportEnv(gym.Env):
    def __init__(self, board_size, zoffset, reward, penalty):
        super().__init__()

        self.width = board_size
        self.height = board_size
        self.obs_shape = (self.height, self.width, 2)

        self.action_space = spaces.Discrete(self.width)
        self.observation_space = spaces.Box(0, 255, self.obs_shape, dtype=np.uint8)

        self.zoffset = zoffset
        self.reward = reward
        self.penalty = penalty

    def is_valid_action(self, action):
        if (
            self.model[self.action_row, action] == 255
            or self.support[self.action_row, action] == 255
        ):
            return False
        else:
            return True

    def step(self, action):
        if not self.is_valid_action(action):
            return self.obs(), -9999.0, False, {}

        self.support[self.action_row, action] = 255

        if self.update_action_row():
            if self.action_row == self.height:
                return self.obs(), self.reward, True, {}
            else:
                return self.obs(), self.reward, False, {}
        else:
            return self.obs(), -self.penalty / 100, False, {}

    def _remove_noise(self):
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
            self.model, connectivity=4
        )

        total_area = np.array([stats[i, cv2.CC_STAT_AREA] for i in range(1, retval)])
        idx = np.argmax(total_area) + 1
        for i in range(self.height):
            for j in range(self.width):
                if idx != labels[i, j]:
                    self.model[i, j] = 0

    def reset(self):
        while True:
            self.model = np.zeros((self.obs_shape[0], self.obs_shape[1]), dtype=np.uint8)
            self.support = np.zeros((self.obs_shape[0], self.obs_shape[1]), dtype=np.uint8)

            max_sample = int((self.height-self.zoffset) * self.width)
            sample_num = np.random.randint(1, max_sample)
            samples = np.random.choice(max_sample, size=sample_num, replace=False)
            for sample in samples:
                row = sample // self.width
                col = sample % self.width
                self.model[row, col] = 255

            self._remove_noise()

            self.action_row = 1
            self.update_action_row()
            if self.action_row != self.height:
                break

        return self.obs()

    def _is_stable(self, row):
        upper_support = self.support[row, :]
        lower_support = self.support[row + 1, :]

        support_size = 0
        if support_size != 0:
            upper_support = ndimage.binary_dilation(
                upper_support, iterations=support_size
            )
            lower_support = ndimage.binary_dilation(
                lower_support, iterations=support_size
            )

        upper = np.logical_or(self.model[row, :], upper_support)
        lower = np.logical_or(self.model[row + 1, :], lower_support)

        # intersection = np.logical_and(upper, lower)
        # dilation_size = 1
        # dilated = ndimage.binary_dilation(intersection, mask=upper, iterations=dilation_size)

        dilated = ndimage.binary_dilation(lower)

        res = np.logical_and(upper, np.logical_not(dilated))
        supported = not res.any()
        return supported

    def update_action_row(self):
        action_row_change = False
        while self.action_row < self.height:
            if self._is_stable(self.action_row - 1):
                self.action_row += 1
                action_row_change = True
            else:
                break

        return action_row_change

    def obs(self):
        return np.stack([self.model, self.support], axis=2)

    def render(self, close=False):
        print("\n", "=" * 15, "\n", sep="")
        for i in range(self.height):
            str = ""
            for j in range(self.width):
                if self.model[i, j] == 255:
                    str += "@"
                elif self.support[i, j] == 255:
                    str += "*"
                else:
                    str += " "
            print(str)

if __name__ == "__main__":
    env = SupportEnv(5)
    env = LegalActionWrapper(env)
    env.reset()
    env.render()
