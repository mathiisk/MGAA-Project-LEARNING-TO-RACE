"""
set_variant.py — copy the right config into TmrlData before launching.

Run this ONCE before starting server/trainer/worker:
    python set_variant.py A
    python set_variant.py C

Then start your 3 terminals as normal:
    python src/run_experiment.py --variant A --role server
    python src/run_experiment.py --variant A --role trainer
    python src/run_experiment.py --variant A --role worker --track rewards/reward_track1.pkl
"""
import sys
import json
import shutil
from pathlib import Path

variant = sys.argv[1].upper() if len(sys.argv) > 1 else None
if variant not in ("A", "B", "C", "D"):
    print("Usage: python set_variant.py [A|B|C|D]")
    sys.exit(1)

project_root = Path(__file__).resolve().parent
src = project_root / "config" / f"config_variant{variant}.json"
dst = Path.home() / "TmrlData" / "config" / "config.json"

if not src.exists():
    print(f"ERROR: {src} not found")
    sys.exit(1)

dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(src, dst)

with open(dst) as f:
    cfg = json.load(f)

print(f"OK — copied config_variant{variant}.json to {dst}")
print(f"   RUN_NAME  : {cfg['RUN_NAME']}")
print(f"   INTERFACE : {cfg['ENV']['RTGYM_INTERFACE']}")
print(f"   MAX_EPOCHS: {cfg['MAX_EPOCHS']}")
print(f"   SAVE_MODEL_EVERY: {cfg['SAVE_MODEL_EVERY']}")
print()
print(f"Now start your 3 terminals:")
print(f"  python src/run_experiment.py --variant {variant} --role server")
print(f"  python src/run_experiment.py --variant {variant} --role trainer")
print(f"  python src/run_experiment.py --variant {variant} --role worker --track rewards/reward_trackN.pkl")
