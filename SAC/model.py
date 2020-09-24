import gym

import torch as th
import torch.nn as nn

import numpy as np

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


class ResidualBlock(nn.Module):
    def __init__(self, n_channel):
        super(ResidualBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(n_channel, n_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(),
            nn.Conv2d(n_channel, n_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_channel),
        )

        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.net(x)
        x = residual + x
        return self.activation(x)


class ResFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, n_channel, n_block):
        super(ResFeatureExtractor, self).__init__(observation_space, 1)

        # self.conv_in = nn.Sequential(
        #     nn.Conv2d(
        #         observation_space.shape[0],
        #         n_channel,
        #         kernel_size=3,
        #         padding=1,
        #         bias=False,
        #     ),
        #     nn.BatchNorm2d(n_channel),
        #     nn.ReLU(),
        # )

        # layers = []
        # for _ in range(n_block):
        #     layers.append(ResidualBlock(n_channel))
        # self.res_block = nn.Sequential(*layers)

        # self.conv_in.apply(init_weights)
        # self.res_block.apply(init_weights)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                observation_space.shape[0],
                n_channel,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                n_channel,
                n_channel,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                n_channel,
                n_channel,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                n_channel,
                n_channel,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(),
        )

        self.conv1.apply(init_weights)
        self.conv2.apply(init_weights)
        self.conv3.apply(init_weights)
        self.conv4.apply(init_weights)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # x = self.conv_in(observations)
        # return self.res_block(x)
        x = self.conv1(observations)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class Res_Q_ValueHead(nn.Module):
    def __init__(self, feature_shape, n_actions):
        super(Res_Q_ValueHead, self).__init__()

        self.net1 = nn.Sequential(
            nn.Conv2d(feature_shape[0], 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
        )

        n_flatten = self._get_conv_val_size(feature_shape)
        self.net2 = nn.Sequential(nn.Linear(n_flatten, n_actions))

        self.net1.apply(init_weights)
        self.net2.apply(init_weights)

    def _get_conv_val_size(self, shape):
        with th.no_grad():
            o = self.net1(th.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x, actions):
        qvalues = self.net2(self.net1(x))
        return th.gather(qvalues, 1, actions.unsqueeze(-1))


class Res_PolicyHead(nn.Module):
    def __init__(self, feature_shape, n_actions):
        super(Res_PolicyHead, self).__init__()

        self.net1 = nn.Sequential(
            nn.Conv2d(feature_shape[0], 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
        )

        n_flatten = self._get_conv_policy_size(feature_shape)
        self.net2 = nn.Sequential(nn.Linear(n_flatten, n_actions))

        self.net1.apply(init_weights)
        self.net2.apply(init_weights)

    def _get_conv_policy_size(self, shape):
        with th.no_grad():
            o = self.net1(th.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        return self.net2(self.net1(x))