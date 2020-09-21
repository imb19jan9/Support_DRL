from ppo import ppo
from env import SupportEnv, ImageToPyTorch, ScaledFloatFrame
from model import ResActorCritic

from spinup.utils.run_utils import setup_logger_kwargs

env_fn = lambda: ScaledFloatFrame(ImageToPyTorch(SupportEnv()))

ac_kwargs = dict(n_channel=64, n_block=6)

logger_kwargs = dict(output_dir="./data", exp_name="exp1")
logger_kwargs = setup_logger_kwargs(exp_name="8x8", data_dir="./data", datestamp=True)
ppo(
    env_fn=env_fn,
    actor_critic=ResActorCritic,
    ac_kwargs=ac_kwargs,
    target_kl=0.12,
    epochs=1000,
    logger_kwargs=logger_kwargs,
)
