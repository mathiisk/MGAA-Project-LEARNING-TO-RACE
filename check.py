import pickle
import numpy as np

with open("rewards/reward_track1.pkl", "rb") as f:
    data = pickle.load(f)

print(f"Total points: {len(data)}")

# distance between consecutive points
diffs = np.diff(data, axis=0)
dists = np.linalg.norm(diffs, axis=1)
print(f"Avg dist between points: {dists.mean():.3f} units")
print(f"Min: {dists.min():.3f}  Max: {dists.max():.3f}")

# total track length
print(f"Total track length: {dists.sum():.1f} units")

# points advanced per step at typical speeds
# at 20 steps/sec, how far does the car travel per step?
print(f"\nAt 100 km/h (~27 m/s), per step (0.05s): {27*0.05:.2f}m")
print(f"Points that covers: {27*0.05/dists.mean():.1f} points/step")
print(f"Reward per step at 100km/h: {27*0.05/dists.mean()/100:.4f}")