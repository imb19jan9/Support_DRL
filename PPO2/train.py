from ppo import ppo
from env import SupportEnv_v0, ImageToPyTorch, ScaledFloatFrame
from model import ResActorCritic

from spinup.utils.run_utils import setup_logger_kwargs

seed = 0
env_fn = lambda: ScaledFloatFrame(ImageToPyTorch(SupportEnv_v0()))

ac_kwargs = dict(n_channel=128, n_block=6)

logger_kwargs = dict(output_dir="./data", exp_name="exp1")
logger_kwargs = setup_logger_kwargs(exp_name="board8_nc128_nb6", seed=seed, data_dir="./data", datestamp=True)
ppo(
    env_fn=env_fn,
    actor_critic=ResActorCritic,
    ac_kwargs=ac_kwargs,
    seed = seed,
    target_kl=0.12,
    logger_kwargs=logger_kwargs,
)
