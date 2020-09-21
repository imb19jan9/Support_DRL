import gym

import torch as th
import torch.nn as nn

import numpy as np

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

RES_NUM_BLOCK = 8
RES_NUM_FILTERS = 256

def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(RES_NUM_FILTERS, RES_NUM_FILTERS, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(RES_NUM_FILTERS),
            nn.ReLU(),
            nn.Conv2d(RES_NUM_FILTERS, RES_NUM_FILTERS, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(RES_NUM_FILTERS)
        )

        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.net(x)
        x = residual + x
        return self.activation(x)

class DualResNet(BaseFeaturesExtractor):
        def __init__(self, observation_space: gym.spaces.Box, features_dim: int = RES_NUM_FILTERS):
            super(DualResNet, self).__init__(observation_space, features_dim)

            self.conv_in = nn.Sequential(
                nn.Conv2d(observation_space.shape[0], RES_NUM_FILTERS, kernel_size=8, stride=4, bias=False),
                nn.BatchNorm2d(RES_NUM_FILTERS),
                nn.ReLU()
            )

            layers = []
            for _ in range(RES_NUM_BLOCK):
                layers.append(ResidualBlock())
            self.res_block = nn.Sequential(*layers)

            self.conv_in.apply(init_weights)
            self.res_block.apply(init_weights)


        def forward(self, observations: th.Tensor) -> th.Tensor:
            x = self.conv_in(observations)
            return self.res_block(x)

class Res_ValueHead(nn.Module):
    def __init__(self, feature_shape, hidden_dim=64):
        super(Res_ValueHead, self).__init__()

        self.net1 = nn.Sequential(
            nn.Conv2d(RES_NUM_FILTERS, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten()
        )

        n_flatten = self._get_conv_val_size(feature_shape)
        self.net2 = nn.Sequential(
            nn.Linear(n_flatten, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )

        self.net1.apply(init_weights)
        self.net2.apply(init_weights)

    def _get_conv_val_size(self, shape):
        with th.no_grad():
            o = self.net1(th.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        return self.net2(self.net1(x))

class Res_PolicyHead(nn.Module):
    def __init__(self, feature_shape, n_actions):
        super(Res_PolicyHead, self).__init__()
        
        self.net1 = nn.Sequential(
            nn.Conv2d(RES_NUM_FILTERS, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten()
        )

        n_flatten = self._get_conv_policy_size(feature_shape)
        self.net2 = nn.Sequential(
            nn.Linear(n_flatten, n_actions)
        )

        self.net1.apply(init_weights)
        self.net2.apply(init_weights)

    def _get_conv_policy_size(self, shape):
        with th.no_grad():
            o = self.net1(th.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        return self.net2(self.net1(x))

class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(NatureCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        self.cnn.apply(init_weights)
        self.linear.apply(init_weights)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

class Cnn_ValueHead(nn.Module):
    def __init__(self, features_dim):
        super(Cnn_ValueHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(features_dim, 1)
        )

        self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)

class Cnn_PolicyHead(nn.Module):
    def __init__(self, features_dim, n_actions):
        super(Cnn_PolicyHead, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(features_dim, n_actions)
        )

        self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)