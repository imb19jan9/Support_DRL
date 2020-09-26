from datetime import datetime

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
    n_envs = 8
    wrapper_class = lambda env: ScaledFloatFrame(
        ImageToPyTorch(LegalActionWrapper(env))
    )
    env_kwargs = dict(board_size=30, zoffset=8, reward=0.1, penalty=0.005)
    features_extractor_kwargs = dict(n_channel=128, n_block=8)
    optimizer_kwargs = dict(weight_decay=0)
    ppo_kwargs = dict(
        learning_rate=3e-4,
        n_steps=256,
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
        verbose=1,
        seed=seed
    )
    policy_kwargs = dict(
        valuehead_hidden=256,
        features_extractor_class=ResFeatureExtractor,
        features_extractor_kwargs=features_extractor_kwargs,
        optimizer_kwargs=optimizer_kwargs,
    )

    env = make_vec_env(
        SupportEnv,
        n_envs,
        seed=seed,
        wrapper_class=wrapper_class,
        env_kwargs=env_kwargs,
    )

    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H:%M:%S")
    model = PPO(
        MyActorCriticPolicy,
        env,
        **ppo_kwargs,
        tensorboard_log=f"runs/{date_time}",
        policy_kwargs=policy_kwargs,
    )
    print(model.policy)
    model.save("./logs/rl_model_0_steps")

    checkpoint_callback = CheckpointCallback(
        save_freq=int(1e5), save_path="./logs/", name_prefix="rl_model"
    )
    model.learn(
        total_timesteps=int(5e6),
        reset_num_timesteps=False,
        callback=checkpoint_callback,
    )
