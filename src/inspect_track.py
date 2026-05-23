"""
Prints statistics about point density, track length, and how many centerline
points the car advances per environment step at typical racing speeds. Useful
for tuning the waypoint stride and reward scaling before a training run.

Usage:
    python src/inspect_track.py --track reward_track1.pkl
    python src/inspect_track.py --track reward_track2.pkl
"""

import argparse
import pickle
import numpy as np
from pathlib import Path

REWARDS_DIR = Path(__file__).resolve().parent.parent / "rewards"


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a track centerline .pkl file and print useful statistics."
    )
    parser.add_argument(
        "--track",
        required=True,
        help="Track .pkl filename inside the rewards/ folder (e.g. reward_track1.pkl)."
    )
    args = parser.parse_args()

    # Always resolve relative to the rewards/ folder — no need to type the full path.
    track_path = REWARDS_DIR / args.track
    if not track_path.exists():
        print(f"ERROR: track file not found: {track_path}")
        return

    with open(track_path, "rb") as f:
        data = pickle.load(f)

    print(f"Track file : {track_path}")
    print(f"Total points: {len(data)}")

    # Measure spacing between consecutive centerline points.
    # This tells us the resolution of the reward signal and how to set waypoint stride.
    diffs = np.diff(data, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    print(f"Avg dist between points: {dists.mean():.3f} units")
    print(f"Min: {dists.min():.3f}  Max: {dists.max():.3f}")

    # Sum of all segment lengths gives the full track distance in game units.
    # TMRL's coordinate system is in metres, so this is directly interpretable.
    print(f"Total track length: {dists.sum():.1f} units")

    # Estimate how many centerline points the car covers per environment step.
    # TMRL runs at 20 steps/sec (0.05s per step) by default.
    # This tells us that if stride=10 in get_lookahead_waypoints(), how far ahead are we actually looking?
    # And how coarse/fine the progress reward is at race speeds.
    print(f"\nAt 100 km/h (~27 m/s), per step (0.05s): {27*0.05:.2f}m")
    print(f"Points that covers: {27*0.05/dists.mean():.1f} points/step")
    print(f"Reward per step at 100km/h: {27*0.05/dists.mean()/100:.4f}")


if __name__ == "__main__":
    main()