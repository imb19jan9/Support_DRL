from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from functools import partial
import gym
import numpy as np

import torch as th
import torch.nn as nn
from torch.distributions import Categorical

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import Distribution, CategoricalDistribution

from model import ResFeatureExtractor, Res_PolicyHead, Res_ValueHead

class MyActorCriticPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable,
        device: Union[th.device, str] = "auto",
        use_sde: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = ResFeatureExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            device,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=False,
        )

        self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.action_dist = CategoricalDistribution(self.action_space.n)

        self._build(lr_schedule)

    def _get_data(self) -> Dict[str, Any]:
        data = dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            use_sde=self.use_sde,
            lr_schedule=self._dummy_schedule,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            features_extractor_class=self.features_extractor_class,
            features_extractor_kwargs=self.features_extractor_kwargs,
        )
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        raise NotImplementedError("do not use reset_noise")

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        zero_input = th.zeros(1, *self.observation_space.shape)
        feature_shape = self.features_extractor(zero_input).squeeze(0).shape
        self.action_net = Res_PolicyHead(feature_shape, self.action_space.n)
        self.value_net = Res_ValueHead(feature_shape, hidden_dim=256)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: (th.Tensor) Observation
        :param deterministic: (bool) Whether to sample or use deterministic actions
        :return: (Tuple[th.Tensor, th.Tensor, th.Tensor]) action, value and log probability of the action
        """
        
        latent = self._get_latent(obs)
        values = self.value_net(latent)

        legal_actions = self._legal_actions(obs)
        distribution = self._get_action_dist_from_latent(latent, legal_actions)
        
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob
    
    def _get_latent(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.

        :param obs: (th.Tensor) Observation
        :return: (Tuple[th.Tensor, th.Tensor, th.Tensor]) Latent codes
            for the actor, the value function and for gSDE function
        """
        return self.features_extractor(obs)

    def _get_action_dist_from_latent(self, latent_pi, legal_actions) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: (th.Tensor) Latent code for the actor
        :param latent_sde: (Optional[th.Tensor]) Latent code for the gSDE exploration function
        :return: (Distribution) Action distribution
        """
        logits = self.action_net(latent_pi)
        logits_exp = (th.exp(logits) + 1e-8) * legal_actions
        probs = logits_exp / th.sum(logits_exp, dim=1).unsqueeze(-1)
        self.action_dist.distribution = Categorical(probs=probs)
        return self.action_dist

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation: (th.Tensor)
        :param deterministic: (bool) Whether to use stochastic or deterministic actions
        :return: (th.Tensor) Taken action according to the policy
        """
        latent = self._get_latent(observation)
        legal_actions = self._legal_actions(observation)
        distribution = self._get_action_dist_from_latent(latent, legal_actions)
        return distribution.get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: (th.Tensor)
        :param actions: (th.Tensor)
        :return: (th.Tensor, th.Tensor, th.Tensor) estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        latent = self._get_latent(obs)
        legal_actions = self._legal_actions(obs)
        distribution = self._get_action_dist_from_latent(latent, legal_actions)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent)
        return values, log_prob, distribution.entropy()

    def _legal_actions(self, obs):
        legal_actions = th.sum(obs[:,-1,:,:], dim=1)
        return legal_actions