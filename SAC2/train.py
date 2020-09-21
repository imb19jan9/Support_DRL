from spinup import sac_pytorch
from sac import sac
from env import SupportEnv, ImageToPyTorch, ScaledFloatFrame
from model import ResActorCritic

from spinup.utils.run_utils import setup_logger_kwargs

seed = 0
env_fn = lambda: ScaledFloatFrame(ImageToPyTorch(SupportEnv()))

ac_kwargs = dict(n_channel=128, n_block=6)

logger_kwargs = dict(output_dir="./data", exp_name="exp1")
logger_kwargs = setup_logger_kwargs(exp_name="8x8", seed=seed, data_dir="./data", datestamp=True)
sac(
    env_fn=env_fn,
    actor_critic=ResActorCritic,
    ac_kwargs=ac_kwargs,
    seed=seed,
    epochs=1000,
    logger_kwargs=logger_kwargs,
)
