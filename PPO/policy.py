from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from functools import partial
import gym
import numpy as np

import torch as th
import torch.nn as nn
from torch.distributions import Categorical

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import CategoricalDistribution

from model import DualResNet, NatureCNN, Res_ValueHead, Res_PolicyHead, RES_NUM_FILTERS, Cnn_ValueHead, Cnn_PolicyHead

class MyActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        device: Union[th.device, str] = "auto",
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = False,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = DualResNet,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = False,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            device,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        super()._build(lr_schedule)

        self.action_dist = CategoricalDistribution(self.action_space.n)

        if isinstance(self.features_extractor, DualResNet):
            temp = th.zeros(1, *self.observation_space.shape)
            feature_shape = self.features_extractor(temp).squeeze(0).shape
            self.value_net = Res_ValueHead(feature_shape)
            self.action_net = Res_PolicyHead(feature_shape, self.action_space.n)
        if isinstance(self.features_extractor, NatureCNN):
            features_dim = self.features_extractor.features_dim
            self.value_net = Cnn_ValueHead(features_dim)
            self.action_net = Cnn_PolicyHead(features_dim, self.action_space.n)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation: (th.Tensor)
        :param deterministic: (bool) Whether to use stochastic or deterministic actions
        :return: (th.Tensor) Taken action according to the policy
        """
        legal_actions = self.legal_actions(observation)
        
        features = self.features_extractor(observation)
        logits = self.action_net(features)

        legal_logits_exp = th.exp(logits) * legal_actions
        legal_probs = legal_logits_exp / th.sum(legal_logits_exp, dim=1).unsqueeze(-1)

        self.action_dist.distribution = Categorical(probs=legal_probs)
        return self.action_dist.get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: (th.Tensor)
        :param actions: (th.Tensor)
        :return: (th.Tensor, th.Tensor, th.Tensor) estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        legal_actions = self.legal_actions(obs)

        features = self.features_extractor(obs)
        logits = self.action_net(features)
        values = self.value_net(features)

        legal_logits_exp = th.exp(logits) * legal_actions
        legal_probs = legal_logits_exp / th.sum(legal_logits_exp, dim=1).unsqueeze(-1)

        self.action_dist.distribution = Categorical(probs=legal_probs)
        log_prob = self.action_dist.log_prob(actions)

        return values, log_prob, self.action_dist.entropy()
    
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: (th.Tensor) Observation
        :param deterministic: (bool) Whether to sample or use deterministic actions
        :return: (Tuple[th.Tensor, th.Tensor, th.Tensor]) action, value and log probability of the action
        """
        legal_actions = self.legal_actions(obs)

        features = self.features_extractor(obs)
        logits = self.action_net(features)
        values = self.value_net(features)

        legal_logits_exp = th.exp(logits) * legal_actions
        legal_probs = legal_logits_exp / th.sum(legal_logits_exp, dim=1).unsqueeze(-1)

        self.action_dist.distribution = Categorical(probs=legal_probs)
        actions = self.action_dist.get_actions(deterministic=deterministic)
        log_prob = self.action_dist.log_prob(actions)

        return actions, values, log_prob

    def legal_actions(self, obs):
        legal_actions = th.sum(obs[:,-1,:,:], dim=1)
        return legal_actions