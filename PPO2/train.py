from ppo import ppo
from env import SupportEnv_v1, ImageToPyTorch, ScaledFloatFrame
from model import ResActorCritic

from spinup.utils.run_utils import setup_logger_kwargs

seed = 0
env_fn = lambda: ScaledFloatFrame(ImageToPyTorch(SupportEnv_v1()))

ac_kwargs = dict(n_channel=128, n_block=8)

logger_kwargs = dict(output_dir="./data", exp_name="exp1")
logger_kwargs = setup_logger_kwargs(exp_name="v1_board8_nc128_nb8", seed=seed, data_dir="./data", datestamp=False)
ppo(
    env_fn=env_fn,
    actor_critic=ResActorCritic,
    ac_kwargs=ac_kwargs,
    seed = seed,
    target_kl=0.12,
    logger_kwargs=logger_kwargs,
)
