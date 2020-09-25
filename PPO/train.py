from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from env import SupportEnv, LegalActionWrapper, ImageToPyTorch, ScaledFloatFrame
from policy import MyActorCriticPolicy
from model import ResFeatureExtractor


def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func


if __name__ == "__main__":
    seed = 0
    n_channel = 128
    n_block = 2
    board_size = 8

    n_envs = 8
    env_kwargs=dict(board_size=board_size, reward=0.5)
    env = make_vec_env(
        SupportEnv,
        n_envs,
        seed=seed,
        wrapper_class=lambda env: ScaledFloatFrame(ImageToPyTorch(LegalActionWrapper(env))),
        env_kwargs=env_kwargs,
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
        learning_rate=1e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.05,
        tensorboard_log=f"runs/board{board_size}_nc{n_channel}_nb{n_block}_seed{seed}",
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
    )

    print(model.policy)

    checkpoint_callback = CheckpointCallback(
        save_freq=int(1e5), save_path="./logs/", name_prefix="rl_model"
    )

    model.learn(
        total_timesteps=int(1e7), reset_num_timesteps=False, callback=checkpoint_callback
    )
