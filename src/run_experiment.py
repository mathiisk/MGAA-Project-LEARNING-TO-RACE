"""
Usage (3 terminals per variant, same structure for all):
    python src/run_experiment.py --role server
    python src/run_experiment.py --role trainer
    python src/run_experiment.py --role worker

Results
-------
    results/<RUN_NAME>.csv — written by the worker, one row per episodes
    
We can pass --wandb to the trainer to enable WandB logging.
The worker always writes to CSV, regardless of WandB.
config.json needs setup of wandb project and entity for WandB logging.
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR   = PROJECT_ROOT / "config"
RESULTS_DIR  = PROJECT_ROOT / "results"


def _set_config() -> str:
    """
    Copy config/config.json to ~/TmrlData/config/config.json — the one path
    TMRL always reads, regardless of version or env vars.
    Returns the RUN_NAME so we can name the results CSV correctly.
    """
    import shutil

    src = CONFIG_DIR / "config.json"
    if not src.exists():
        print(f"ERROR: config file not found: {src}")
        print(f"  Expected: {CONFIG_DIR}/config.json")
        sys.exit(1)

    tmrl_data = Path.home() / "TmrlData"
    tmrl_data.mkdir(parents=True, exist_ok=True)
    dest = tmrl_data / "config" / "config.json"
    dest.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy(src, dest)

    with open(src) as f:
        cfg = json.load(f)

    run_name = cfg["RUN_NAME"]
    steps_per_epoch = (
        cfg["ROUNDS_PER_EPOCH"]
        * cfg["TRAINING_STEPS_PER_ROUND"]
        / cfg["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]
    )
    budget = int(cfg["MAX_EPOCHS"] * steps_per_epoch)

    print(f"[run_experiment] config    -> copied {src.name} to {dest}")
    print(f"[run_experiment] run_name  -> {run_name}")
    print(f"[run_experiment] budget    -> ~{budget:,} env steps (MAX_EPOCHS={cfg['MAX_EPOCHS']})")
    print()
    return run_name


def _install_reward_file(track_pkl: str):
    """
    Copy the chosen track .pkl into TMRL's reward slot.

    TMRL ALWAYS reads the reward signal from a single fixed path:
        <TMRL_DATA>/reward/reward.pkl
    It has no concept of per-track files. So before the worker starts we
    copy the requested track file into that slot. This makes --track the
    single source of truth: the reward signal is guaranteed to come from
    the correct track file.

    Must run BEFORE the worker imports/starts tmrl.
    """
    import shutil

    src = Path(track_pkl)
    if not src.is_absolute():
        src = PROJECT_ROOT / track_pkl
    if not src.exists():
        print(f"ERROR: track reward file not found: {src}")
        sys.exit(1)

    tmrl_data = Path.home() / "TmrlData"
    dest_dir = tmrl_data / "reward"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "reward.pkl"

    shutil.copy(src, dest)
    print(f"[run_experiment] reward    -> copied {src.name} into {dest}")


def make_env_cls(run_name):
    """
    Return a no-arg env class for TMRL to instantiate.
    Wraps the base TMRL env with StepCounterWrapper so results CSVs
    are always written.

    TMRL expects env_cls to be a callable that takes no arguments and returns
    a ready-to-use environment. We achieve this by defining an inner class
    whose __new__ builds the full wrapper stack and returns it.

    Args:
        run_name: Passed to StepCounterWrapper to name the results CSV.

    Returns:
        A class whose instantiation returns the fully wrapped environment.
    """
    import tmrl.config.config_objects as cfg_obj
    from envs.step_counter import StepCounterWrapper

    base_cls = cfg_obj.ENV_CLS
    results_dir = str(RESULTS_DIR)

    class EnvCls:
        def __new__(cls):
            env = base_cls()
            return StepCounterWrapper(env, run_name=run_name, results_dir=results_dir)

    return EnvCls


# ---------------------------------------------------------------------------
# Server and Trainer delegate directly to `python -m tmrl` — they never
# instantiate the env, so there's no reason to fight TMRL's internal API.
# config.json is already in place, so they pick up the right config.
#
# Only the Worker needs our custom env_cls, because that's the only process
# that actually calls env_cls() to collect experience.
# ---------------------------------------------------------------------------

def run_server():
    """
    Delegate to `python -m tmrl --server`.
    ~/TmrlData/config/config.json is already set by _set_config().
    """
    import subprocess
    subprocess.run(
        [sys.executable, "-m", "tmrl", "--server"],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        check=True,
    )


def run_trainer(wandb=False):
    """
    Delegate to `python -m tmrl --trainer`.
    ~/TmrlData/config/config.json is already set by _set_config(),
    so tmrl picks it up automatically.
    """
    import subprocess
    cmd = [sys.executable, "-m", "tmrl", "--trainer"]
    if wandb:
        cmd.append("--wandb")
    subprocess.run(
        cmd,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        check=True,
    )


def run_worker(env_cls):
    """
    Build and run a TMRL RolloutWorker with our custom env.

    The worker is the only process that actually steps through the environment
    and sends experience to the server. Everything else (obs space, reward,
    action space) is determined by the base TMRL config.

    Args:
        env_cls: The env class returned by make_env_cls().
    """
    import tmrl.config.config_constants as cfg
    import tmrl.config.config_objects as cfg_obj
    from tmrl.networking import RolloutWorker
    from tmrl.util import partial

    worker = RolloutWorker(
        env_cls=env_cls,
        actor_module_cls=partial(cfg_obj.POLICY),
        sample_compressor=cfg_obj.SAMPLE_COMPRESSOR,
        obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
        device="cpu",
        server_ip=cfg.SERVER_IP_FOR_WORKER,
        max_samples_per_episode=cfg.TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"],
        model_path=cfg.MODEL_PATH_WORKER,
        crc_debug=False,
    )
    worker.run()


def main():
    parser = argparse.ArgumentParser(
        description="Launcher for the raw LiDAR SAC experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--role", choices=["server", "trainer", "worker"], required=True,
                        help="Which TMRL process to start.")
    parser.add_argument("--track", default="rewards/reward_track1.pkl",
                        help="Track reward .pkl copied into TMRL's reward slot before the worker starts.")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable WandB logging for the trainer.")
    args = parser.parse_args()

    run_name = _set_config()

    print(f"[run_experiment] role={args.role} | obs=81-dim LiDAR\n")

    if args.role == "server":
        run_server()

    elif args.role == "trainer":
        run_trainer(wandb=args.wandb)

    elif args.role == "worker":
        # Install the reward file for this track BEFORE building the env.
        # This is what makes --track actually control the reward signal.
        _install_reward_file(args.track)
        env_cls = make_env_cls(run_name=run_name)
        run_worker(env_cls)


if __name__ == "__main__":
    main()