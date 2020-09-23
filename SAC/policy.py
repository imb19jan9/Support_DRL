from stable_baselines3.common.policies import BaseModel, BasePolicy


from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn as nn
from torch.distributions import Categorical

from stable_baselines3.common.distributions import Distribution, CategoricalDistribution
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, create_sde_features_extractor, register_policy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN, create_mlp

from model import ResFeatureExtractor, Res_PolicyHead, Res_Q_ValueHead

class DiscreteCritic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: (gym.spaces.Space) Obervation space
    :param action_space: (gym.spaces.Space) Action space
    :param net_arch: ([int]) Network architecture
    :param features_extractor: (nn.Module) Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: (int) Number of features
    :param activation_fn: (Type[nn.Module]) Activation function
    :param normalize_images: (bool) Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param device: (Union[th.device, str]) Device on which the code should run.
    :param n_critics: (int) Number of critic networks to create.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: nn.Module,
        device: Union[th.device, str] = "auto",
        n_critics: int = 2,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=False,
            device=device,
        )

        self.n_critics = n_critics
        self.q_networks = []

        zero_input = th.zeros(1, *self.observation_space.shape)
        feature_shape = self.features_extractor(zero_input).squeeze(0).shape
        for idx in range(n_critics):
            q_net = Res_Q_ValueHead(feature_shape, self.action_space.n)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        with th.no_grad():
            features = self.features_extractor(obs)
        return tuple(q_net(features, actions) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.features_extractor(obs)
        return self.q_networks[0](features, actions)

class Actor(BasePolicy):
    """
    Actor network (policy) for SAC.

    :param observation_space: (gym.spaces.Space) Obervation space
    :param action_space: (gym.spaces.Space) Action space
    :param net_arch: ([int]) Network architecture
    :param features_extractor: (nn.Module) Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: (int) Number of features
    :param activation_fn: (Type[nn.Module]) Activation function
    :param use_sde: (bool) Whether to use State Dependent Exploration or not
    :param log_std_init: (float) Initial value for the log standard deviation
    :param full_std: (bool) Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param sde_net_arch: ([int]) Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: (bool) Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: (float) Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: (bool) Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param device: (Union[th.device, str]) Device on which the code should run.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: nn.Module,
        device: Union[th.device, str] = "auto",
    ):
        super(Actor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=False,
            device=device,
        )    

        self.action_dist = CategoricalDistribution(self.action_space.n)

        zero_input = th.zeros(1, *self.observation_space.shape)
        feature_shape = self.features_extractor(zero_input).squeeze(0).shape
        self.action_net = Res_PolicyHead(feature_shape, self.action_space.n)

    def _get_data(self) -> Dict[str, Any]:
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            features_extractor=self.features_extractor,
        )

    def get_std(self) -> th.Tensor:
        raise NotImplementedError('Do not use get_std function')

    def reset_noise(self, batch_size: int = 1) -> None:
        raise NotImplementedError('Do not use reset_noise function')

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        raise NotImplementedError('Do not use get_action_dist_params function')

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        latent = self._get_latent(obs)
        legal_actions = self._legal_actions(obs)
        distribution = self._get_action_dist_from_latent(latent, legal_actions)
        return distribution.get_actions(deterministic=deterministic)

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        latent = self._get_latent(obs)
        legal_actions = self._legal_actions(obs)
        distribution = self._get_action_dist_from_latent(latent, legal_actions)
        
        actions = distribution.get_actions()
        log_prob = distribution.log_prob(actions)
        
        return actions, log_prob

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.forward(observation, deterministic)

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

    def _legal_actions(self, obs):
        legal_actions = th.sum(obs[:,-1,:,:], dim=1)
        return legal_actions

class SACDPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: (gym.spaces.Space) Observation space
    :param action_space: (gym.spaces.Space) Action space
    :param lr_schedule: (callable) Learning rate schedule (could be constant)
    :param net_arch: (Optional[List[int]]) The specification of the policy and value networks.
    :param device: (str or th.device) Device on which the code should run.
    :param activation_fn: (Type[nn.Module]) Activation function
    :param use_sde: (bool) Whether to use State Dependent Exploration or not
    :param log_std_init: (float) Initial value for the log standard deviation
    :param sde_net_arch: ([int]) Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: (bool) Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: (float) Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: (Type[BaseFeaturesExtractor]) Features extractor to use.
    :param features_extractor_kwargs: (Optional[Dict[str, Any]]) Keyword arguments
        to pass to the feature extractor.
    :param normalize_images: (bool) Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: (Type[th.optim.Optimizer]) The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: (Optional[Dict[str, Any]]) Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: (int) Number of critic networks to create.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable,
        device: Union[th.device, str] = "auto",
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
    ):
        super(SACDPolicy, self).__init__(
            observation_space,
            action_space,
            device,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        # Create shared features extractor
        self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "features_extractor": self.features_extractor,
            "device": device,
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update({"n_critics": n_critics})

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None

        self._build(lr_schedule)

    def _build(self, lr_schedule: Callable) -> None:
        self.actor = self.make_actor()
        self.critic = self.make_critic()
        self.critic_target = self.make_critic()
        
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        # Do not optimize the shared feature extractor with the critic loss
        # otherwise, there are gradient computation issues
        # Another solution: having duplicated features extractor but requires more memory and computation
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        critic_parameters = [param for name, param in self.critic.named_parameters() if "features_extractor" not in name]
        self.critic.optimizer = self.optimizer_class(critic_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_data(self) -> Dict[str, Any]:
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            n_critics=self.critic_kwargs["n_critics"],
            lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            features_extractor_class=self.features_extractor_class,
            features_extractor_kwargs=self.features_extractor_kwargs,
        )

    def reset_noise(self, batch_size: int = 1) -> None:
        raise NotImplementedError('Do not use reset_noise function')

    def make_actor(self) -> Actor:
        return Actor(**self.actor_kwargs)

    def make_critic(self) -> DiscreteCritic:
        return DiscreteCritic(**self.critic_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic)