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
    Copy the variant config to ~/TmrlData/config.json — the one path
    TMRL always reads, regardless of version or env vars.
    Returns the RUN_NAME so we can name the results CSV correctly.
    """
    import shutil

    src = CONFIG_DIR / f"config_variant{variant}.json"
    if not src.exists():
        print(f"ERROR: config file not found: {src}")
        print(f"  Expected: {CONFIG_DIR}/config_variant{{A,B,C,D}}.json")
        sys.exit(1)

    # TMRL always reads from ~/TmrlData/config.json
    tmrl_data = Path.home() / "TmrlData"
    tmrl_data.mkdir(parents=True, exist_ok=True)
    dest = tmrl_data / "config.json"

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
    single source of truth: the reward signal and (for C/D) the waypoint
    observations are guaranteed to come from the same track.

    Must run BEFORE the worker imports/starts tmrl.
    """
    import shutil

    src = Path(track_pkl)
    if not src.is_absolute():
        src = PROJECT_ROOT / track_pkl
    if not src.exists():
        print(f"ERROR: track reward file not found: {src}")
        sys.exit(1)

    # TMRL_DATA is always ~/TmrlData — TMRL doesn't expose it as a constant
    tmrl_data = Path.home() / "TmrlData"
    dest_dir = tmrl_data / "reward"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "reward.pkl"

    shutil.copy(src, dest)
    print(f"[run_experiment] reward    -> copied {src.name} into {dest}")


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

class CarPosWrapper:
    """StepCounterWrapper that also updates car_pos_ref on every step."""
    def __init__(self, env, run_name, results_dir, car_pos_ref):
        from envs.step_counter import StepCounterWrapper
        self._sc = StepCounterWrapper(env, run_name=run_name, results_dir=results_dir)
        self._ref = car_pos_ref
        self.observation_space = self._sc.observation_space
        self.action_space = self._sc.action_space

    def reset(self, **kwargs):
        return self._sc.reset(**kwargs)

    def step(self, action):
        result = self._sc.step(action)
        # Update car position from game state
        try:
            state = self._sc.env.unwrapped.interface.game_state
            self._ref[0] = [float(state[3]), float(state[5])]
        except Exception:
            pass
        return result

    def __getattr__(self, name):
        return getattr(self._sc, name)


def make_env_cls(variant, track_pkl, n_waypoints, stride, run_name, car_pos_ref=None):
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
                return CarPosWrapper(env, run_name=run_name, results_dir=results_dir,
                                     car_pos_ref=car_pos_ref)

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
    """
    Delegate to `python -m tmrl --server`.
    ~/TmrlData/config.json is already set to the right variant config.
    """
    import subprocess, sys
    subprocess.run(
        [sys.executable, "-m", "tmrl", "--server"],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        check=True,
    )


def run_trainer(env_cls):
    """
    Delegate to `python -m tmrl --trainer`.
    ~/TmrlData/config.json is already set to the right variant config
    by _set_config(), so tmrl picks it up automatically.
    """
    import subprocess, sys
    subprocess.run(
        [sys.executable, "-m", "tmrl", "--trainer"],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        check=True,
    )

def make_sample_compressor(variant, track_pkl, n_waypoints, stride, car_pos_ref):
    import tmrl.config.config_objects as cfg_obj
    base = cfg_obj.SAMPLE_COMPRESSOR
    if variant not in ("C", "D"):
        return base
    from envs.augmented_lidar_env import load_centerline, smooth_racing_line, get_lookahead_waypoints
    import numpy as np
    track = track_pkl if Path(track_pkl).is_absolute() else str(PROJECT_ROOT / track_pkl)
    centerline = load_centerline(track)
    line = smooth_racing_line(centerline) if variant == "D" else centerline
    mid = len(line) // 2

    def compressor(*args):
        # sample is (act, obs, rew, terminated, truncated, info) or similar
        # base compressor converts obs Tuple -> compressed form
        compressed = base(*args)
        car_xz = np.array(car_pos_ref[0]) if car_pos_ref[0] is not None else line[mid, [0, 2]]
        extra = get_lookahead_waypoints(car_xz, line, n_waypoints, stride)
        # append waypoints to the obs portion of the compressed sample
        # compressed[1] is the obs
        obs = compressed[1]
        if isinstance(obs, (tuple, list)):
            obs_flat = np.concatenate([np.array(o).flatten() for o in obs])
        else:
            obs_flat = np.array(obs).flatten()
        obs_aug = np.concatenate([obs_flat, extra]).astype(np.float32)
        return (compressed[0], obs_aug, *compressed[2:])
    return compressor

def make_obs_preprocessor(variant, track_pkl, n_waypoints, stride, car_pos_ref):
    """
    car_pos_ref is a list [xz] shared between the env and preprocessor.
    The env updates it on every step; the preprocessor reads it.
    """
    import tmrl.config.config_objects as cfg_obj
    base_preprocessor = cfg_obj.OBS_PREPROCESSOR
    if variant not in ("C", "D"):
        return base_preprocessor
    from envs.augmented_lidar_env import load_centerline, smooth_racing_line, get_lookahead_waypoints
    import numpy as np
    track = track_pkl if Path(track_pkl).is_absolute() else str(PROJECT_ROOT / track_pkl)
    centerline = load_centerline(track)
    line = smooth_racing_line(centerline) if variant == "D" else centerline
    mid = len(line) // 2

    def preprocessor(obs):
        obs = base_preprocessor(obs)   # may still be a tuple
        if isinstance(obs, (tuple, list)):
            flat = np.concatenate([np.array(o).flatten() for o in obs])
        else:
            flat = np.array(obs).flatten()
        car_xz = np.array(car_pos_ref[0]) if car_pos_ref[0] is not None else line[mid, [0, 2]]
        extra = get_lookahead_waypoints(car_xz, line, n_waypoints, stride)
        return np.concatenate([flat, extra]).astype(np.float32)
    return preprocessor


def run_worker(env_cls, variant="A", track_pkl="rewards/reward_track1.pkl", n_waypoints=5, stride=10, car_pos_ref=None):
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
                        help="Track reward .pkl. Copied into TMRL's reward slot "
                             "(all variants) AND used for waypoint obs (C/D).")
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
        # Install the reward file for this track BEFORE building the env.
        # This is what makes --track actually control the reward signal.
        _install_reward_file(args.track)
        # Only the worker needs the custom env — build it here
        car_pos_ref = [None]   # shared mutable: env writes, preprocessor reads
        env_cls = make_env_cls(
            variant=args.variant,
            track_pkl=args.track,
            n_waypoints=args.n_waypoints,
            stride=args.stride,
            run_name=run_name,
            car_pos_ref=car_pos_ref,
        )
        run_worker(env_cls, args.variant, args.track, args.n_waypoints, args.stride, car_pos_ref)


if __name__ == "__main__":
    main()
