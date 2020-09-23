import gym
from gym import spaces
import numpy as np
from scipy import ndimage
from collections import deque


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.float32
        )

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class SupportEnv_v0(gym.Env):
    metadata = {"render.modes": ["console"]}

    def __init__(self):
        self.width = 8
        self.height = 8
        self.obs_shape = (self.height, self.width, 3)

        self.action_space = spaces.Discrete(self.width)
        self.observation_space = spaces.Box(0, 255, self.obs_shape, dtype=np.uint8)

    def step(self, action):
        if not self._is_valid_action(action):
            return self.obs(), -1.0, False, {}

        self.support[self.action_row, action] = 255

        if self.update_action_row():
            if self.action_row == self.height:
                return self.obs(), 1, True, {}
            else:
                return self.obs(), 1, False, {}
        else:
            return self.obs(), -0.01, False, {}

    def reset(self):
        self.model = np.zeros((self.obs_shape[0], self.obs_shape[1]), dtype=np.uint8)
        self.support = np.zeros((self.obs_shape[0], self.obs_shape[1]), dtype=np.uint8)

        max_sample = int(self.height * self.width * 0.5)
        sample_num = np.random.randint(1, max_sample)
        samples = np.random.choice(max_sample, size=sample_num, replace=False)
        for sample in samples:
            row = sample // self.width
            col = sample % self.width
            self.model[row, col] = 255

        self.action_row = 1
        self.update_action_row()

        return self.obs()

    def _is_valid_action(self, action):
        if (
            self.model[self.action_row, action] == 255
            or self.support[self.action_row, action] == 255
        ):
            return False
        else:
            return True

    def render(self, mode="human", close=False):
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

    def update_action_row(self):
        action_row_change = False
        while self.action_row < self.height:
            if self._is_stable(self.action_row - 1):
                self.action_row += 1
                action_row_change = True
            else:
                break

        return action_row_change

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

    def _legal_action_image(self):
        img = np.zeros((self.obs_shape[0], self.obs_shape[1]), dtype=np.uint8)

        if self.action_row == self.height:
            img[0, :] = 255
            return img

        model_empty = self.model[self.action_row, :] == 0
        support_empty = self.support[self.action_row, :] == 0
        empty = np.logical_and(model_empty, support_empty)
        legal_actions = np.nonzero(empty)[0]

        img[self.action_row, legal_actions] = 255
        return img

    def obs(self):
        state = np.zeros(self.obs_shape, dtype=np.uint8)
        state[:, :, 0] = self.model
        state[:, :, 1] = self.support
        state[:, :, 2] = self._legal_action_image()
        return state

class SupportEnv_v1(SupportEnv_v0):
    def __init__(self):
        super().__init__()

    def step(self, action):
        if not self._is_valid_action(action):
            return self.obs(), -1.0, False, {}

        self.support[self.action_row, action] = 255

        if self.update_action_row():
            if self.action_row == self.height:
                return self.obs(), 1, True, {}
            else:
                return self.obs(), 1, False, {}
        else:
            return self.obs(), 0, False, {}