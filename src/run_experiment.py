"""
run_experiment.py — unified launcher for all observation-space variants.

Each variant has its own config file in config/. The launcher sets
TMRL_CONFIG_PATH before TMRL loads anything, so every variant gets
isolated model weights, checkpoints, replay memory, and results CSV.

Usage (3 terminals per variant, same structure for all):
    python src/run_experiment.py --variant A --role server
    python src/run_experiment.py --variant A --role trainer
    python src/run_experiment.py --variant A --role worker

    python src/run_experiment.py --variant C --role server --track rewards/reward_track1.pkl
    python src/run_experiment.py --variant C --role trainer --track rewards/reward_track1.pkl
    python src/run_experiment.py --variant C --role worker  --track rewards/reward_track1.pkl

Variants
--------
    A  —  raw LiDAR only       (81-dim obs)
    B  —  image / TM20FULL     (image stacked obs)
    C  —  LiDAR + waypoints    (91-dim obs, centerline lookahead)
    D  —  LiDAR + racing line  (91-dim obs, smoothed centerline)

Step budget
-----------
    Controlled by MAX_EPOCHS in each config file.
    Default: 20 epochs = ~100,000 env steps. Trainer stops automatically.

Results
-------
    results/<RUN_NAME>.csv — written by the worker, one row per episode.
    After all variants finish: python src/plot_results.py

Offline sanity check (no TrackMania needed):
    python src/envs/augmented_lidar_env.py --variant C --track rewards/reward_track1.pkl
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR   = PROJECT_ROOT / "config"
RESULTS_DIR  = PROJECT_ROOT / "results"


def _set_config(variant: str) -> str:
    """
    Point TMRL_CONFIG_PATH at the right config file BEFORE any tmrl import.
    Returns the RUN_NAME so we can name the results CSV correctly.
    """
    config_path = CONFIG_DIR / f"config_variant{variant}.json"
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}")
        print(f"  Expected: {CONFIG_DIR}/config_variant{{A,B,C,D}}.json")
        sys.exit(1)

    os.environ["TMRL_CONFIG_PATH"] = str(config_path)

    with open(config_path) as f:
        cfg = json.load(f)

    run_name = cfg["RUN_NAME"]
    steps_per_epoch = (
        cfg["ROUNDS_PER_EPOCH"]
        * cfg["TRAINING_STEPS_PER_ROUND"]
        / cfg["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]
    )
    budget = int(cfg["MAX_EPOCHS"] * steps_per_epoch)

    print(f"[run_experiment] config    -> {config_path}")
    print(f"[run_experiment] run_name  -> {run_name}")
    print(f"[run_experiment] budget    -> ~{budget:,} env steps (MAX_EPOCHS={cfg['MAX_EPOCHS']})")
    print()
    return run_name


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def make_env_cls(variant: str, track_pkl: str, n_waypoints: int, stride: int, run_name: str):
    """
    Return a no-arg env class for TMRL to instantiate.
    All variants get StepCounterWrapper so results CSVs are always written.
    Variants C/D additionally get WaypointAugmentedEnv inside that.
    """
    import tmrl.config.config_objects as cfg_obj
    from envs.augmented_lidar_env import WaypointAugmentedEnv
    from envs.step_counter import StepCounterWrapper

    base_cls = cfg_obj.ENV_CLS
    results_dir = str(RESULTS_DIR)

    if variant in ("A", "B"):
        class EnvCls:
            def __new__(cls):
                env = base_cls()
                return StepCounterWrapper(env, run_name=run_name, results_dir=results_dir)
    else:
        class EnvCls:
            def __new__(cls):
                env = base_cls()
                env = WaypointAugmentedEnv(env, track_pkl=track_pkl,
                                           variant=variant,
                                           n_waypoints=n_waypoints,
                                           stride=stride)
                return StepCounterWrapper(env, run_name=run_name, results_dir=results_dir)

    return EnvCls


# ---------------------------------------------------------------------------
# Role runners
#
# Server and Trainer delegate directly to `python -m tmrl` — they never
# instantiate the env, so there's no reason to fight TMRL's internal API.
# TMRL_CONFIG_PATH is already set, so they pick up the right config.
#
# Only the Worker needs our custom env_cls, because that's the only process
# that actually calls env_cls() to collect experience.
# ---------------------------------------------------------------------------

def run_server():
    """Delegate to tmrl's own server — it reads everything from TMRL_CONFIG_PATH."""
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "tmrl", "--server"], check=True)


def run_trainer(env_cls):
    """
    Delegate to tmrl's own trainer — the trainer only does gradient updates
    on replay buffer data and never instantiates the env directly.
    Our custom env_cls is not needed here; the network input size is inferred
    from the data already in the buffer (collected by the worker).
    """
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "tmrl", "--trainer"], check=True)


def run_worker(env_cls):
    import tmrl.config.config_constants as cfg
    import tmrl.config.config_objects as cfg_obj
    from tmrl.networking import RolloutWorker
    from tmrl.util import partial

    worker = RolloutWorker(
        env_cls=env_cls,
        actor_module_cls=partial(cfg_obj.POLICY),
        sample_compressor=cfg_obj.SAMPLE_COMPRESSOR,
        device="cpu",
        server_ip=cfg.SERVER_IP_FOR_WORKER,
        max_samples_per_episode=cfg.TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"],
        obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
        model_path=cfg.MODEL_PATH_WORKER,
        crc_debug=False,
    )
    worker.run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified launcher for all observation-space ablation variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--variant", choices=["A", "B", "C", "D"], required=True)
    parser.add_argument("--role", choices=["server", "trainer", "worker"], required=True)
    parser.add_argument("--track", default="rewards/reward_track1.pkl",
                        help="Reward .pkl path. Used for variants C and D only.")
    parser.add_argument("--n_waypoints", type=int, default=5)
    parser.add_argument("--stride", type=int, default=10,
                        help="Waypoint stride in raw-point units (10 = 1 m apart)")
    args = parser.parse_args()

    # Set TMRL_CONFIG_PATH before any tmrl import — subprocess inherits it too
    run_name = _set_config(args.variant)

    obs_note = {
        "A": "81-dim LiDAR",
        "B": "image stack (TM20FULL)",
        "C": f"{81 + args.n_waypoints * 2}-dim LiDAR+waypoints",
        "D": f"{81 + args.n_waypoints * 2}-dim LiDAR+racing-line",
    }[args.variant]
    print(f"[run_experiment] variant={args.variant} | role={args.role} | obs={obs_note}\n")

    # Server and trainer delegate to `python -m tmrl` — they never touch the env.
    # TMRL_CONFIG_PATH is inherited by the subprocess so the right config is used.
    if args.role == "server":
        run_server()
    elif args.role == "trainer":
        run_trainer(env_cls=None)
    elif args.role == "worker":
        # Only the worker needs the custom env — build it here
        env_cls = make_env_cls(
            variant=args.variant,
            track_pkl=args.track,
            n_waypoints=args.n_waypoints,
            stride=args.stride,
            run_name=run_name,
        )
        run_worker(env_cls)


if __name__ == "__main__":
    main()
