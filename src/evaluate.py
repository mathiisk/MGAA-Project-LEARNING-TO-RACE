"""
Runs a trained TMRL agent for N episodes (default 20) on a given track
and reports all metrics needed for the paper's evaluation section.

Reported metrics (per-run CSV + summary table printed to stdout):
  - Lap completion rate  (fraction of episodes that reach the finish line)
  - Mean episode length  in steps for COMPLETED laps (lower = faster)
  - Wall-clock lap time  in seconds for COMPLETED laps
  - Fastest lap time     (wall-clock, seconds)

How it works
------------
TMRL's `--test` flag runs one episode using the weights at
    ~/TmrlData/weights/<RUN_NAME>.tmod
and then exits.  We replicate that behaviour in-process so we can loop
over episodes without restarting Python, and capture every episode's stats.

The script does NOT need the server or trainer to be running.

Usage
-----
Basic (Track 5, 20 rollouts, reads RUN_NAME from config/config.json):
    python evaluate.py

Choose a different model weights file:
    python evaluate.py --weights ~/TmrlData/weights/Curriculum.tmod

Choose a different track (generalisation test):
    python evaluate.py --track rewards/reward_track_held_out.pkl

Full example - baseline model, held-out track, 20 rollouts:
    python evaluate.py \\
        --weights ~/TmrlData/weights/NoCurriculum.tmod \\
        --track   rewards/reward_held_out.pkl \\
        --runs    20 \\
        --label   baseline_generalisation

Output
------
  results/eval_<label>.csv - one row per episode
  stdout - summary table
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path


# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR   = PROJECT_ROOT / "config"
RESULTS_DIR  = PROJECT_ROOT / "results"
REWARDS_DIR  = PROJECT_ROOT / "rewards"
SRC_DIR      = PROJECT_ROOT / "src"

# add src/ to path so we can import project modules
sys.path.insert(0, str(SRC_DIR))


# helpers 
def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


def install_config(config_path: Path):
    """Copy project config into the slot TMRL always reads from."""
    dest = Path.home() / "TmrlData" / "config" / "config.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, dest)
    print(f"[eval] config     -> {dest}")


def install_weights(weights_path: Path, run_name: str):
    """
    Copy the chosen .tmod into TMRL's weights slot for <run_name>.
    TMRL's worker/test mode loads:  ~/TmrlData/weights/<RUN_NAME>.tmod
    """
    dest_dir = Path.home() / "TmrlData" / "weights"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{run_name}.tmod"
    shutil.copy(weights_path, dest)
    print(f"[eval] weights -> {dest}")


def install_track(track_path: Path):
    """Copy chosen track .pkl into TMRL's single reward slot."""
    dest_dir = Path.home() / "TmrlData" / "reward"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "reward.pkl"
    shutil.copy(track_path, dest)
    print(f"[eval] track      -> {dest}")


def is_lap_complete(reward: float, end_of_track_reward: float) -> bool:
    """
    TMRL signals track completion by giving a large positive terminal reward
    (END_OF_TRACK, default 6.0).  Any episode whose last step reward is
    >= end_of_track_reward is counted as a completed lap.
    We use a small margin (0.5) to tolerate floating-point noise.
    """
    return reward >= end_of_track_reward - 0.5


def run_one_episode(env, actor, obs_preprocessor, end_of_track_reward: float) -> dict:
    """
    Run a single episode to completion (done=True) or until the env's
    ep_max_length is reached.

    Returns a dict with:
        completed       bool - True if the agent reached the finish line
        episode_length  int - total steps taken
        episode_reward  float - total undiscounted reward
        wall_time_s     float - wall-clock seconds for this episode
        last_reward     float - reward on the final step (used for completion check)
    """
    obs, _ = env.reset() if hasattr(env.reset, '__code__') and \
             env.reset.__code__.co_argcount > 1 else (env.reset(), None)

    # handle both old (obs,) and new (obs, info) reset signatures
    if isinstance(obs, tuple):
        obs, _info = obs

    total_reward = 0.0
    steps = 0
    last_reward = 0.0
    t0 = time.time()

    while True:
        # pre-process observation the same way TMRL's test mode does
        obs_proc = obs_preprocessor(obs) if obs_preprocessor is not None else obs
        import torch, numpy as np
        if isinstance(obs_proc, (list, tuple)):
            obs_proc = [torch.tensor(o, dtype=torch.float32).unsqueeze(0) if isinstance(o, np.ndarray) else o for o in obs_proc]
        elif isinstance(obs_proc, np.ndarray):
            obs_proc = torch.tensor(obs_proc, dtype=torch.float32).unsqueeze(0)

        # actor forward pass
        import torch
        with torch.no_grad():
            action = actor.act(obs_proc, test=True)

        result = env.step(action)

        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result

        last_reward = float(reward)
        total_reward += last_reward
        steps += 1

        if done:
            break

    wall_time = time.time() - t0
    completed = is_lap_complete(last_reward, end_of_track_reward)

    return {
        "completed":      completed,
        "episode_length": steps,
        "episode_reward": round(total_reward, 4),
        "wall_time_s":    round(wall_time, 3),
        "last_reward":    round(last_reward, 4),
    }


# main loop
def evaluate(
    weights_path: Path | None,
    track_path:   Path,
    n_runs:       int,
    label:        str,
    config_path:  Path,
):
    # load config
    cfg = load_config(config_path)
    run_name = cfg["RUN_NAME"]
    end_of_track_reward = cfg["ENV"]["REWARD_CONFIG"]["END_OF_TRACK"]

    print(f"\n[eval] ========================================")
    print(f"[eval] label      : {label}")
    print(f"[eval] run_name   : {run_name}")
    print(f"[eval] weights    : {weights_path}")
    print(f"[eval] track      : {track_path}")
    print(f"[eval] n_runs     : {n_runs}")
    print(f"[eval] completion reward threshold: {end_of_track_reward - 0.5:.1f}")
    print(f"[eval] ========================================\n")

    # install config, weights and track into TMRL's data directory
    install_config(config_path)
    if weights_path is not None:
        install_weights(weights_path, run_name)
    install_track(track_path)

    # build the TMRL environment and load the policy
    import tmrl.config.config_constants as tmrl_cfg
    import tmrl.config.config_objects   as cfg_obj

    print("[eval] building environment …")
    env = cfg_obj.ENV_CLS()

    print("[eval] loading actor …")
    import torch
    actor = cfg_obj.POLICY(env.observation_space, env.action_space)
    model_path = tmrl_cfg.MODEL_PATH_WORKER          # ~/TmrlData/weights/<RUN_NAME>.tmod
    if not Path(model_path).exists():
        print(f"[eval] ERROR: model weights not found at {model_path}")
        print(f"       Make sure --weights points to the correct .tmod file.")
        sys.exit(1)
    actor.load_state_dict(torch.load(model_path, map_location="cpu"))
    actor.eval()
    print(f"[eval] loaded weights from {model_path}\n")

    obs_preprocessor = cfg_obj.OBS_PREPROCESSOR

    # run N episodes
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"eval_{label}.csv"

    results = []
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run", "completed", "episode_length",
            "episode_reward", "wall_time_s", "last_reward"
        ])

        for i in range(1, n_runs + 1):
            print(f"[eval] run {i:>2}/{n_runs} … ", end="", flush=True)
            stats = run_one_episode(env, actor, obs_preprocessor, end_of_track_reward)
            results.append(stats)

            status = "COMPLETE" if stats["completed"] else "did not finish"
            print(
                f"{status} | "
                f"steps={stats['episode_length']:>4} | "
                f"reward={stats['episode_reward']:>8.2f} | "
                f"time={stats['wall_time_s']:.1f}s"
            )

            writer.writerow([
                i,
                int(stats["completed"]),
                stats["episode_length"],
                stats["episode_reward"],
                stats["wall_time_s"],
                stats["last_reward"],
            ])
            f.flush()

    env.close()

    # compute and print summary
    print(f"\n[eval] ── Summary ({'':─<42}")
    n_complete = sum(r["completed"] for r in results)
    completion_rate = n_complete / n_runs

    completed_runs = [r for r in results if r["completed"]]
    if completed_runs:
        mean_steps    = sum(r["episode_length"] for r in completed_runs) / len(completed_runs)
        mean_time     = sum(r["wall_time_s"]    for r in completed_runs) / len(completed_runs)
        fastest_time  = min(r["wall_time_s"]    for r in completed_runs)
        fastest_steps = min(r["episode_length"] for r in completed_runs)
    else:
        mean_steps = mean_time = fastest_time = fastest_steps = float("nan")

    print(f"  Completion rate   : {n_complete}/{n_runs}  ({completion_rate*100:.1f}%)")
    if completed_runs:
        print(f"  Mean steps (comp) : {mean_steps:.1f}  steps")
        print(f"  Mean time  (comp) : {mean_time:.2f} s")
        print(f"  Fastest lap time  : {fastest_time:.2f} s  ({fastest_steps} steps)")
    else:
        print("  No completed laps — completion metrics not applicable.")

    print(f"\n[eval] Full results written to: {csv_path}")
    print(f"[eval] ==================================================\n")

    # also write a one-line summary CSV
    summary_path = RESULTS_DIR / "eval_summary.csv"
    write_header = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "label", "n_runs", "n_complete",
                "completion_rate", "mean_steps_completed",
                "mean_wall_time_s_completed", "fastest_lap_s",
                "weights", "track",
            ])
        writer.writerow([
            label, n_runs, n_complete,
            round(completion_rate, 4),
            round(mean_steps,   2) if completed_runs else "",
            round(mean_time,    3) if completed_runs else "",
            round(fastest_time, 3) if completed_runs else "",
            str(weights_path),
            str(track_path),
        ])
    print(f"[eval] Summary row appended to: {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained TMRL agent for N episodes and report paper metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--weights",
        help=(
            "Path to the .tmod weights file to evaluate. "
            "Defaults to ~/TmrlData/weights/<RUN_NAME>.tmod (i.e. whatever is currently installed)."
        ),
    )
    parser.add_argument(
        "--track",
        default=str(REWARDS_DIR / "reward_track1.pkl"),
        help="Path to the track reward .pkl file. Default: rewards/reward_track1.pkl",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of evaluation rollouts (default: 20).",
    )
    parser.add_argument(
        "--label",
        default=None,
        help=(
            "Label used in the output CSV filename (eval_<label>.csv). "
            "Defaults to <RUN_NAME>_<track_stem>."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(CONFIG_DIR / "config.json"),
        help="Path to config.json (default: config/config.json).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        sys.exit(1)

    cfg      = load_config(config_path)
    run_name = cfg["RUN_NAME"]

    # resolve weights path
    if args.weights:
        weights_path = Path(args.weights).expanduser()
        if not weights_path.exists():
            print(f"ERROR: weights file not found: {weights_path}")
            sys.exit(1)
    else:
        weights_path = None

    # resolve track path
    track_path = Path(args.track)
    if not track_path.is_absolute():
        track_path = PROJECT_ROOT / track_path
    if not track_path.exists():
        print(f"ERROR: track file not found: {track_path}")
        sys.exit(1)

    # build label
    label = args.label or f"{run_name}_{track_path.stem}"

    evaluate(
        weights_path=weights_path,
        track_path=track_path,
        n_runs=args.runs,
        label=label,
        config_path=config_path,
    )


if __name__ == "__main__":
    main()