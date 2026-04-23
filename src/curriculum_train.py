import subprocess
import shutil
import time
from pathlib import Path

TMRL_DATA = Path("C:/Users/Matiss/TmrlData")
REWARD_DIR = TMRL_DATA / "rewards"       # put your reward_trackN.pkl files here
CHECKPOINT_DIR = TMRL_DATA / "checkpoints"

# Define your curriculum stages
STAGES = [
    {
        "name": "STRAIGHT-ROAD",
        "reward_file": REWARD_DIR / "reward_track1.pkl",
        "episodes": 500,          # train for this many episodes then evaluate
        "success_threshold": 0.8, # mean reward fraction to advance (tune this)
    },
    {
        "name": "TURNS-ROAD",
        "reward_file": REWARD_DIR / "reward_track2.pkl",
        "episodes": 800,
        "success_threshold": 0.75,
    },
    {
        "name": "CURVY-ROAD",
        "reward_file": REWARD_DIR / "reward_track3.pkl",
        "episodes": 1200,
        "success_threshold": 0.0, # final stage, always complete
    },
]

def switch_track(reward_file: Path):
    """Swap in the reward file for the new track."""
    dest = TMRL_DATA / "reward.pkl"
    shutil.copy(reward_file, dest)
    print(f"Switched reward to: {reward_file.name}")

def run_stage(stage: dict):
    print(f"\n=== Starting stage: {stage['name']} ===")
    switch_track(stage["reward_file"])
    
    # You manually start --server, --trainer, --worker in separate terminals
    # This script just signals you when to stop and advance
    input(f"Start your 3 tmrl terminals now, then press Enter when done training stage '{stage['name']}'...")
    print(f"Stage {stage['name']} complete. Weights are saved automatically by tmrl.")

if __name__ == "__main__":
    for stage in STAGES:
        run_stage(stage)
    print("\nCurriculum complete!")