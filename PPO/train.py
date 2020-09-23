from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env

from env import SupportEnv_v0, ImageToPyTorch, ScaledFloatFrame
from policy import MyActorCriticPolicy
from model import ResFeatureExtractor

if __name__ == "__main__":
    seed = 0
    n_channel=64
    n_block=6

    n_envs = 1
    env = make_vec_env(
        SupportEnv_v0,
        n_envs,
        seed=seed,
        wrapper_class=lambda env: ScaledFloatFrame(ImageToPyTorch(env)),
    )

    optimizer_kwargs = dict(weight_decay=5e-4)
    features_extractor_kwargs = dict(n_channel=n_channel, n_block=n_block)
    policy_kwargs = dict(
        features_extractor_class=ResFeatureExtractor,
        features_extractor_kwargs=features_extractor_kwargs,
        optimizer_kwargs=optimizer_kwargs,
    )

    model = PPO(
        MyActorCriticPolicy,
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.05,
        tensorboard_log=f"runs/v0_board8_nc{n_channel}_nb{n_block}_seed{seed}",
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
    )

    print(model.policy)

    model.learn(total_timesteps=1e6)
    model.save("ppo_model")
