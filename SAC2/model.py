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

    def forward(self, x):
        return self.net2(self.net1(x))


class ResQCritic(nn.Module):
    def __init__(self, feature_shape, act_dim):
        super().__init__()

        self.net1 = nn.Sequential(
            nn.Conv2d(feature_shape[0], 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            Flatten(),
        )

        n_flatten = self._get_conv_val_size(feature_shape)
        self.net2 = nn.Sequential(nn.Linear(n_flatten, act_dim))

        self.net1.apply(init_weights)
        self.net2.apply(init_weights)

    def _get_conv_val_size(self, shape):
        with torch.no_grad():
            o = self.net1(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        return self.net2(self.net1(x))


class ResTwinQCritic(nn.Module):
    def __init__(self, feature_shape, act_dim):
        super().__init__()

        self.q1 = ResQCritic(feature_shape, act_dim)
        self.q2 = ResQCritic(feature_shape, act_dim)

    def forward(self, x):
        return self.q1(x), self.q2(x)


class ResActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, n_channel=64, n_block=6):
        super().__init__()

        obs_dim = observation_space.shape

        self.feature_extractor = ResFeatureExtractor(obs_dim, n_channel, n_block)
        feature_shape = n_channel, obs_dim[1], obs_dim[2]
        self.actor = ResCategoricalActor(feature_shape, action_space.n)
        self.q_critic = ResTwinQCritic(feature_shape, action_space.n)

    def act(self, obs):
        features = self.feature_extractor(obs)
        legal_actions = self.legal_actions(obs)
        logits = self.actor(features, legal_actions)
        action = torch.argmax(logits, dim=1, keepdim=True)
        return action

    def sample(self, obs):
        features = self.feature_extractor(obs)
        legal_actions = self.legal_actions(obs)
        return self.action_probs_logp(features, legal_actions)

    def action_probs_logp(self, features, legal_actions):
        logits = self.actor(features)
        logits_exp = (torch.exp(logits) + 1e-8) * legal_actions
        logit_sum = torch.sum(logits_exp, dim=1).unsqueeze(-1)
        probs = logits_exp / logit_sum
        action_dist = Categorical(probs=probs)
        action = action_dist.sample().view(-1, 1)

        z = (probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(probs + z)

        return action, probs, log_action_probs

    def legal_actions(self, obs):
        return torch.sum(obs[:, -1, :, :], dim=1)
