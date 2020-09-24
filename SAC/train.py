from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from env import SupportEnv_v0, ImageToPyTorch, ScaledFloatFrame
from policy import SACDPolicy
from model import ResFeatureExtractor
from sacd import SACD

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
        learning_rate=linear_schedule(3e-4),
        buffer_size=int(1e6),
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.98,
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

    checkpoint_callback = CheckpointCallback(save_freq=10, save_path='./logs/',
                                         name_prefix='rl_model')
    model.learn(total_timesteps=1e7, reset_num_timesteps=False, callback=checkpoint_callback)
