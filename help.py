"""
Usage:
    python src/help.py
    python src/help.py --script inspect_track
"""

import argparse
from pathlib import Path

REWARDS_DIR = Path(__file__).resolve().parent.parent / "rewards"

SCRIPTS = {
    "run_experiment": {
        "file": "src/run_experiment.py",
        "description": "Main launcher. Starts one of the three TMRL processes for a training run.",
        "usage": [
            "python src/run_experiment.py --role server",
            "python src/run_experiment.py --role trainer",
            "python src/run_experiment.py --role trainer --wandb",
            "python src/run_experiment.py --role worker --track reward_track1.pkl",
        ],
        "arguments": {
            "--role":   "Required. Which process to start: server, trainer, or worker.",
            "--track":  "Track .pkl to load as the reward signal (worker only). Default: reward_track1.pkl.",
            "--wandb":  "Enable WandB logging (trainer only). Requires wandb config in config.json.",
        },
        "notes": (
            "Always start all three roles for a complete training run.\n"
            "  1. server  — relay between trainer and worker\n"
            "  2. trainer — pulls samples from server and updates the policy\n"
            "  3. worker  — drives the car and sends experience to the server\n"
            "Config is read from config/config.json. Edit RUN_NAME and MAX_EPOCHS there."
        ),
    },

    "checkpoint_watcher": {
        "file": "src/checkpoint_watcher.py",
        "description": "Watches TMRL's checkpoint files and saves timestamped copies.",
        "usage": [
            "python src/checkpoint_watcher.py",
            "python src/checkpoint_watcher.py --list",
            "python src/checkpoint_watcher.py --restore checkpoints/RUN_NAME/epoch_0042_reward_187.34",
        ],
        "arguments": {
            "--config":  "Config JSON to read RUN_NAME from. Default: config/config.json.",
            "--list":    "Print all saved checkpoints and exit.",
            "--restore": "Path to a saved epoch folder. Copies files back into TMRL and sets RESET_TRAINING=false.",
        },
        "notes": (
            "Run this in a 4th terminal while server/trainer/worker are running.\n"
            "Polls every 5 seconds. Saves a copy on every new TMRL checkpoint.\n"
            "The best/ folder always contains the highest-reward checkpoint seen so far."
        ),
    },

    "inspect_track": {
        "file": "src/inspect_track.py",
        "description": "Print statistics about a track centerline .pkl file.",
        "usage": [
            "python src/inspect_track.py --track reward_track1.pkl",
        ],
        "arguments": {
            "--track": "Filename of the .pkl inside rewards/. No need to include the folder path.",
        },
        "notes": (
            "Useful before a training run to check point density and set waypoint stride.\n"
            "Available tracks in rewards/:\n"
            + "\n".join(
                f"    {p.name}"
                for p in sorted(REWARDS_DIR.glob("*.pkl"))
            ) if REWARDS_DIR.exists() else "    (rewards/ folder not found)"
        ),
    },
    
    "evaluate": {
    "file": "src/evaluate.py",
    "description": "Evaluate a trained TMRL agent for N episodes and report paper metrics (completion rate, lap time, etc.).",
    "usage": [
        "python src/evaluate.py",
        "python src/evaluate.py --weights ~/TmrlData/weights/Curriculum.tmod",
        "python src/evaluate.py --track rewards/reward_track_held_out.pkl",
        "python src/evaluate.py --weights ~/TmrlData/weights/NoCurriculum.tmod --track rewards/reward_held_out.pkl --runs 20 --label baseline_generalisation",
    ],
    "arguments": {
        "--weights": "Path to a .tmod weights file. Defaults to ~/TmrlData/weights/<RUN_NAME>.tmod (currently installed).",
        "--track":   "Path to a reward track .pkl file. Default: rewards/reward_track1.pkl.",
        "--runs":    "Number of evaluation rollouts. Default: 20.",
        "--label":   "Label for the output CSV filename (eval_<label>.csv). Defaults to <RUN_NAME>_<track_stem>.",
        "--config":  "Path to config.json. Default: config/config.json.",
    },
    "notes": (
        "Does NOT require the server or trainer to be running.\n"
        "Outputs:\n"
        "  results/eval_<label>.csv     — one row per episode\n"
        "  results/eval_summary.csv     — one summary row appended per run\n"
        "Metrics reported (completed laps only):\n"
        "  - Lap completion rate\n"
        "  - Mean episode length (steps)\n"
        "  - Mean wall-clock lap time (seconds)\n"
        "  - Fastest lap time (seconds)"
    ),
},
}


def _divider(char="─", width=60):
    print(char * width)


def _print_script(name, info):
    _divider()
    print(f"  {name}   ({info['file']})")
    _divider()
    print(f"  {info['description']}\n")

    print("  Usage:")
    for line in info["usage"]:
        print(f"    {line}")

    print("\n  Arguments:")
    for arg, desc in info["arguments"].items():
        print(f"    {arg:<16} {desc}")

    if info.get("notes"):
        print("\n  Notes:")
        for line in info["notes"].splitlines():
            print(f"    {line}")
    print()



def main():
    parser = argparse.ArgumentParser(
        description="Show help for all runnable scripts in this project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--script",
        choices=list(SCRIPTS.keys()),
        metavar="NAME",
        help=(
            "Show detailed help for one script. "
            f"Options: {', '.join(SCRIPTS.keys())}"
        ),
    )
    args = parser.parse_args()

    if args.script:
        _print_script(args.script, SCRIPTS[args.script])
    else:
        names = ", ".join(SCRIPTS.keys())
        print("\n  Available scripts — use: python help.py --script NAME")
        print("  NAME options: " + names + "\n")
        _divider("=")
        for name, info in SCRIPTS.items():
            print(f"  {name:<17}  " + f"{info['file']:<24}  " + info["description"])
        _divider("=")
        print()


if __name__ == "__main__":
    main()