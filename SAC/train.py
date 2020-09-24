from stable_baselines3.common.cmd_util import make_vec_env

from env import SupportEnv_v0, ImageToPyTorch, ScaledFloatFrame
from policy import SACDPolicy
from model import ResFeatureExtractor
from sacd import SACD

if __name__ == "__main__":
    seed = 0
    n_channel = 128
    n_block = 12

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

    model = SACD(
        SACDPolicy,
        env,
        learning_rate=3e-4,
        buffer_size=int(1e6),
        learning_starts=100,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        n_episodes_rollout=-1,
        target_update_interval=1,
        target_entropy_ratio=0.9,
        tensorboard_log=f"runs/v0_board8_nc{n_channel}_nb{n_block}_seed{seed}",
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
    )

    print(model.policy)

    for i in range(10):
        model.learn(total_timesteps=1e6, reset_num_timesteps=False)
        model.save(f"ppo_model_{i}")
