"""
Usage:
    python src/plot_results.py
    python src/plot_results.py --results_dir results --smoothing 10 --out results/comparison.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv


VARIANT_STYLE = {
    "SAC_LIDAR_variantA": dict(label="A — LiDAR only",        color="#2196F3", ls="-"),
    "SAC_IMAGE_variantB": dict(label="B — Image (TM20FULL)",  color="#FF9800", ls="--"),
    "SAC_LIDAR_variantC": dict(label="C — LiDAR + waypoints", color="#4CAF50", ls="-"),
    "SAC_LIDAR_variantD": dict(label="D — LiDAR + racing line",color="#9C27B0", ls="-"),
}


def load_csv(path: Path) -> dict:
    """Load a results CSV into lists keyed by column name."""
    data = {col: [] for col in ["wall_time", "env_steps", "episode", "episode_reward", "episode_length"]}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in data:
                data[col].append(float(row[col]))
    return data


def smooth(values: list, window: int) -> np.ndarray:
    """Simple moving average."""
    arr = np.array(values, dtype=float)
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--smoothing", type=int, default=10,
                        help="Moving-average window over episodes. Default: 10")
    parser.add_argument("--out", default="results/comparison.png")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    csv_files = sorted(results_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {results_dir}/")
        print("Run the experiments first, then call this script.")
        return

    print(f"Found {len(csv_files)} result file(s):")
    for f in csv_files:
        print(f"  {f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Observation Space Ablation — SAC on TrackMania", fontsize=13, fontweight="bold")

    ax_reward, ax_throughput = axes

    throughput_data = {}   # run_name -> steps/sec

    for csv_path in csv_files:
        run_name = csv_path.stem
        style = VARIANT_STYLE.get(run_name, dict(label=run_name, color="gray", ls="-"))

        data = load_csv(csv_path)
        if not data["env_steps"]:
            print(f"  {run_name}: empty, skipping")
            continue

        steps    = np.array(data["env_steps"])
        rewards  = smooth(data["episode_reward"], args.smoothing)
        wall     = np.array(data["wall_time"])

        ax_reward.plot(steps, rewards, label=style["label"],
                       color=style["color"], ls=style["ls"], linewidth=1.8, alpha=0.9)

        if wall[-1] > 0:
            throughput_data[style["label"]] = steps[-1] / wall[-1]

        n_eps   = int(data["episode"][-1])
        n_steps = int(steps[-1])
        print(f"  {run_name}: {n_eps} episodes, {n_steps:,} env steps, "
              f"final reward={data['episode_reward'][-1]:.2f}")

    # 1
    ax_reward.set_xlabel("Environment Steps", fontsize=11)
    ax_reward.set_ylabel(f"Episode Reward (smoothed w={args.smoothing})", fontsize=11)
    ax_reward.set_title("Learning Curve", fontsize=11)
    ax_reward.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax_reward.legend(fontsize=9)
    ax_reward.grid(True, alpha=0.3)

    # 2
    if throughput_data:
        labels = list(throughput_data.keys())
        values = [throughput_data[l] for l in labels]
        colors = [VARIANT_STYLE.get(
            next((k for k, v in VARIANT_STYLE.items() if v["label"] == l), ""),
            {}).get("color", "gray") for l in labels]
        bars = ax_throughput.bar(range(len(labels)), values, color=colors, alpha=0.8, edgecolor="white")
        ax_throughput.set_xticks(range(len(labels)))
        ax_throughput.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax_throughput.set_ylabel("Env Steps / Second", fontsize=11)
        ax_throughput.set_title("Training Throughput", fontsize=11)
        ax_throughput.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            ax_throughput.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                               f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
