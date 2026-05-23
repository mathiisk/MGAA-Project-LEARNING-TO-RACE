"""
Watches TMRL's checkpoint and weights files and saves timestamped copies
into your repo's checkpoints/ folder every time TMRL saves.

TMRL saves two files per run:
    ~/TmrlData/checkpoints/<RUN_NAME>.tcpt   — full training state
    ~/TmrlData/weights/<RUN_NAME>.tmod       — model weights (what the worker loads)

Both are watched. A new snapshot is saved whenever either file changes.

Usage (run in a 4th terminal while server/trainer/worker are running):
    python src/checkpoint_watcher.py
    python src/checkpoint_watcher.py --config config/config.json

To restore a checkpoint and resume training:
    python src/checkpoint_watcher.py --restore checkpoints/SAC_4_LIDAR_pretrained/epoch_0042_reward_187.34/

The restore command copies both .tcpt and .tmod back into TmrlData and
ensures RESET_TRAINING=false in ~/TmrlData/config.json.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

POLL_INTERVAL = 5.0  # seconds between checks

def get_run_name(config_path: Path) -> str:
    """
    Read RUN_NAME from a tmrl-style config JSON.

    Args:
        config_path: Path to the config file (e.g. config/config.json).

    Returns:
        The RUN_NAME string, used to locate TMRL's checkpoint and weights files.
    """
    with open(config_path) as f:
        return json.load(f)["RUN_NAME"]


def tmrl_paths(run_name: str) -> dict:
    """
    Return the three file paths TMRL writes for a given run name.
    """
    base = Path.home() / "TmrlData"
    return {
        "tcpt":   base / "checkpoints" / f"{run_name}_t.tcpt",
        "tmod":   base / "weights"     / f"{run_name}.tmod",
        "tmod_t": base / "weights"     / f"{run_name}_t.tmod",
    }


def latest_episode_reward(csv_path: Path) -> float | None:
    """
    Pull the episode_reward from the last data row in our CSV log.
    Used to label checkpoint folders with the reward at save time.

    Returns None if the file doesn't exist, is empty, or can't be parsed,
    we'd rather save an unlabelled checkpoint than crash the watcher.
    """
    if not csv_path.exists():
        return None
    try:
        with open(csv_path) as f:
            lines = f.read().strip().splitlines()
        if len(lines) < 2:
            return None
        # header: wall_time, env_steps, episode, episode_reward, episode_length
        return float(lines[-1].split(",")[3])
    except Exception:
        return None


def best_reward_so_far(save_dir: Path) -> float:
    """
    Read the best reward seen so far from a small tracker file (.best_reward)
    inside the checkpoint directory. Returns -inf if no tracker exists yet,
    so the first real reward always counts as a new best.
    """
    tracker = save_dir / ".best_reward"
    if not tracker.exists():
        return float("-inf")
    try:
        return float(tracker.read_text().strip())
    except Exception:
        return float("-inf")


def write_best_reward(save_dir: Path, reward: float):
    """
    Persist the current best reward to disk so it survives a watcher restart.
    Just writes the float as plain text into .best_reward in the checkpoint dir.
    """
    (save_dir / ".best_reward").write_text(str(reward))


def watch(config_path: Path, project_root: Path):
    """
    Main polling loop. Every POLL_INTERVAL seconds, check whether TMRL has
    written a new checkpoint by comparing the .tcpt file's modification time.
    On a new save: copy both .tcpt and .tmod(s) into a timestamped subfolder,
    and update the best/ folder if this reward beats the previous best.

    Args:
        config_path: Path to config JSON — used to read RUN_NAME.
        project_root: Repo root — checkpoints are saved under project_root/checkpoints/.
    """
    run_name = get_run_name(config_path)
    paths    = tmrl_paths(run_name)
    save_dir = project_root / "checkpoints" / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = project_root / "results" / f"{run_name}.csv"

    print(f"[watcher] run_name : {run_name}")
    print(f"[watcher] watching :")
    print(f"            {paths['tcpt']}")
    print(f"            {paths['tmod']}")
    print(f"[watcher] saving to: {save_dir}")
    print(f"[watcher] poll every {POLL_INTERVAL}s — Ctrl+C to stop")
    print()

    last_mtime = None
    epoch      = 0
    best       = best_reward_so_far(save_dir)
    if best > float("-inf"):
        print(f"[watcher] resuming — previous best reward: {best:.2f}")

    while True:
        time.sleep(POLL_INTERVAL)

        # use .tcpt as the trigger, it's what the trainer writes last
        tcpt   = paths["tcpt"]
        tmod   = paths["tmod"]
        tmod_t = paths["tmod_t"]

        if not tcpt.exists():
            continue  # trainer hasn't saved yet

        mtime = tcpt.stat().st_mtime
        if mtime == last_mtime:
            continue  # no new save

        last_mtime = mtime
        epoch += 1

        reward     = latest_episode_reward(csv_path)
        reward_str = f"{reward:.2f}" if reward is not None else "unknown"

        # save into a subfolder: epoch_0001_reward_45.23/
        epoch_dir = save_dir / f"epoch_{epoch:04d}_reward_{reward_str}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        saved = []
        shutil.copy2(tcpt, epoch_dir / tcpt.name)
        saved.append(tcpt.name)
        if tmod.exists():
            shutil.copy2(tmod, epoch_dir / tmod.name)
            saved.append(tmod.name)
        if tmod_t.exists():
            shutil.copy2(tmod_t, epoch_dir / tmod_t.name)
            saved.append(tmod_t.name)

        print(f"[watcher] epoch {epoch:04d} -> {epoch_dir.name}/  {saved}")

        # track best and save a copy
        if reward is not None and reward > best:
            best = reward
            write_best_reward(save_dir, best)
            best_dir = save_dir / "best"
            best_dir.mkdir(exist_ok=True)
            shutil.copy2(tcpt, best_dir / tcpt.name)
            if tmod.exists():
                shutil.copy2(tmod, best_dir / tmod.name)
            if tmod_t.exists():
                shutil.copy2(tmod_t, best_dir / tmod_t.name)
            print(f"[watcher] *** NEW BEST: {best:.2f} -> saved to checkpoints/{run_name}/best/ ***")
            
            
def restore(epoch_dir: Path):
    """
    Copy a saved checkpoint folder back into TMRL's slots.
    epoch_dir should contain the .tcpt and .tmod files.
    """
    if not epoch_dir.exists() or not epoch_dir.is_dir():
        print(f"ERROR: not a directory: {epoch_dir}")
        return

    # RUN_NAME is the parent folder of the epoch dir
    run_name = epoch_dir.parent.name
    dest     = tmrl_paths(run_name)

    # copy .tcpt
    tcpt_files = list(epoch_dir.glob("*.tcpt"))
    if tcpt_files:
        dest["tcpt"].parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tcpt_files[0], dest["tcpt"])
        print(f"[restore] {tcpt_files[0].name} -> {dest['tcpt']}")
    else:
        print(f"[restore] WARNING: no .tcpt file found in {epoch_dir}")

    # copy .tmod files (worker: RUN_NAME.tmod, trainer: RUN_NAME_t.tmod)
    tmod_files = list(epoch_dir.glob("*.tmod"))
    if tmod_files:
        dest["tmod"].parent.mkdir(parents=True, exist_ok=True)
        for f in tmod_files:
            if f.name.endswith("_t.tmod"):
                shutil.copy2(f, dest["tmod_t"])
                print(f"[restore] {f.name} -> {dest['tmod_t']}")
            else:
                shutil.copy2(f, dest["tmod"])
                print(f"[restore] {f.name} -> {dest['tmod']}")
    else:
        print(f"[restore] WARNING: no .tmod files found in {epoch_dir}")

    # ensure RESET_TRAINING is false
    tmrl_config = Path.home() / "TmrlData" / "config.json"
    if tmrl_config.exists():
        with open(tmrl_config) as f:
            cfg = json.load(f)
        if cfg.get("RESET_TRAINING", True):
            cfg["RESET_TRAINING"] = False
            with open(tmrl_config, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"[restore] set RESET_TRAINING=false in {tmrl_config}")

    print()
    print(f"[restore] Done. Start training with RUN_NAME={run_name} and RESET_TRAINING=false.")


def list_checkpoints(project_root: Path):
    """
    Print a tree of all saved checkpoints under project_root/checkpoints/,
    grouped by run name. Highlights the best/ folder for each run.
    Useful for quickly seeing what's available before a --restore.
    """
    ckpt_root = project_root / "checkpoints"
    if not ckpt_root.exists():
        print("No checkpoints saved yet.")
        return
    for run_dir in sorted(ckpt_root.iterdir()):
        if not run_dir.is_dir():
            continue
        print(f"\n{run_dir.name}/")
        epochs = sorted(d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_"))
        for ep in epochs:
            files = [f.name for f in ep.iterdir() if not f.name.startswith(".")]
            print(f"  {ep.name}/  {files}")
        best = run_dir / "best"
        if best.exists():
            files = [f.name for f in best.iterdir() if not f.name.startswith(".")]
            print(f"  best/  {files}  <- highest reward so far")


def main():
    """
    CLI entry point. Supports three modes:
      - default:    run the file watcher (requires --config or uses the default path)
      - --restore:  copy a saved epoch folder back into TMRL's data directory
      - --list:     print all saved checkpoints and exit
    """
    parser = argparse.ArgumentParser(
        description="Watch TMRL checkpoints and save timestamped copies to your repo.",
    )
    parser.add_argument("--config", default="config/config.json",
                        help="Config JSON to read RUN_NAME from (default: config/config.json)")
    parser.add_argument("--restore", metavar="EPOCH_DIR",
                        help="Restore a saved epoch folder back into TMRL. "
                             "E.g. --restore checkpoints/SAC_4_LIDAR_pretrained/epoch_0042_reward_187.34")
    parser.add_argument("--list", action="store_true",
                        help="List all saved checkpoints and exit.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    if args.list:
        list_checkpoints(project_root)
    elif args.restore:
        epoch_dir = Path(args.restore)
        if not epoch_dir.is_absolute():
            epoch_dir = project_root / epoch_dir
        restore(epoch_dir)
    else:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = project_root / config_path
        if not config_path.exists():
            print(f"ERROR: config not found: {config_path}")
            return
        watch(config_path, project_root)


if __name__ == "__main__":
    main()
