import numpy as np

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


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


class ResFeatureExtractor(nn.Module):
    def __init__(self, obs_dim, n_channel, n_block):
        super(ResFeatureExtractor, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv2d(obs_dim[0], n_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(),
        )

        layers = []
        for _ in range(n_block):
            layers.append(ResidualBlock(n_channel))
        self.res_block = nn.Sequential(*layers)

        self.conv_in.apply(init_weights)
        self.res_block.apply(init_weights)

    def forward(self, x):
        x = self.conv_in(x)
        return self.res_block(x)


class ResCategoricalActor(nn.Module):
    def __init__(self, feature_shape, act_dim):
        super().__init__()

        self.net1 = nn.Sequential(
            nn.Conv2d(feature_shape[0], 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            Flatten(),
        )

        n_flatten = self._get_conv_policy_size(feature_shape)
        self.net2 = nn.Sequential(nn.Linear(n_flatten, act_dim))

        self.net1.apply(init_weights)
        self.net2.apply(init_weights)

    def _get_conv_policy_size(self, shape):
        with torch.no_grad():
            o = self.net1(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def _distribution(self, x, legal_actions):
        logits = self.net2(self.net1(x))
        logits_exp = (torch.exp(logits) + 1e-8) * legal_actions
        logit_sum = torch.sum(logits_exp, dim=1).unsqueeze(-1)
        probs = logits_exp / logit_sum

        return Categorical(probs=probs)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def forward(self, x, legal_actions, act=None):
        pi = self._distribution(x, legal_actions)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class ResCritic(nn.Module):
    def __init__(self, feature_shape, hidden_dim=256):
        super().__init__()

        self.net1 = nn.Sequential(
            nn.Conv2d(feature_shape[0], 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            Flatten(),
        )

        n_flatten = self._get_conv_val_size(feature_shape)
        self.net2 = nn.Sequential(
            nn.Linear(n_flatten, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

        self.net1.apply(init_weights)
        self.net2.apply(init_weights)

    def _get_conv_val_size(self, shape):
        with torch.no_grad():
            o = self.net1(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        return self.net2(self.net1(x))


class ResActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, n_channel=64, n_block=6):
        super().__init__()

        obs_dim = observation_space.shape

        self.feature_extractor = ResFeatureExtractor(obs_dim, n_channel, n_block)
        feature_shape = n_channel, obs_dim[1], obs_dim[2]
        self.pi = ResCategoricalActor(feature_shape, action_space.n)
        self.v = ResCritic(feature_shape, hidden_dim=32)

    def step(self, obs):
        with torch.no_grad():
            features = self.feature_extractor(obs)
            legal_actions = self.legal_actions(obs)
            pi = self.pi._distribution(features, legal_actions)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(features)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]

    def legal_actions(self, obs):
        return torch.sum(obs[:, -1, :, :], dim=1)
