from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy
from stable_baselines3.common.cmd_util import make_vec_env

from env import SupportEnv, ImageToPyTorch, ScaledFloatFrame
from policy import MyActorCriticPolicy
from model import DualResNet, NatureCNN, RES_NUM_BLOCK, RES_NUM_FILTERS

if __name__ == '__main__':
    seed = 0
    n_envs = 1
    env = make_vec_env(
        SupportEnv, n_envs, seed=seed,
        wrapper_class=lambda env: ScaledFloatFrame(ImageToPyTorch(env))
    )

    policy_kwargs = dict(features_extractor_class=NatureCNN,
                         optimizer_kwargs=dict(weight_decay=5e-4),
                         )
    ent_coef = 0.01
    model = PPO(MyActorCriticPolicy,
                env, 
                tensorboard_log='runs/naturecnn',
                policy_kwargs=policy_kwargs,
                verbose=1,
                seed=seed
                )

    print(model.policy)

    timesteps = 10000000
    model.learn(total_timesteps=timesteps)
    model.save("ppo_model")
