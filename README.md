EMPTY
# Curriculum Learning to Race
**MGAIA Group 13 — Leiden University, 2026**

> Curriculum-driven reinforcement learning for TrackMania 2020 using the [tmrl](https://github.com/trackmania-rl/tmrl) framework and Soft Actor-Critic.  

---

## Prerequisites

### 1. Install tmrl
Follow the official guides before doing anything else:
- [Installation guide](https://github.com/trackmania-rl/tmrl/blob/master/readme/Install.md)
- [Getting started](https://github.com/trackmania-rl/tmrl/blob/master/readme/get_started.md)

tmrl requires **TrackMania 2020** (free on Steam and Epic Games), **Windows**, and the **OpenPlanet** plugin. The game must be running before you start any training process.

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

---

## Repository structure

```
├── config/
│   ├── config_baseline.json     # Config for the no-curriculum (direct) agent
│   └── config_curriculum.json   # Config for the curriculum agent
│
├── rewards/                     # Pre-recorded reward trajectory .pkl files
│   ├── track1.pkl  ..  track5.pkl          # Curriculum tracks 1–4 + final track 5
│   └── monaco_held_out.pkl  ..  splits_held_out.pkl  # Held-out evaluation tracks
│
├── trackmania-tracks/           # TrackMania .Map.Gbx files (import into the game)
│   ├── Curriculum-1.Map.Gbx  ..  Curriculum-4.Map.Gbx
│   ├── Track-5.Gbx
│   └── Monaco.Map.Gbx  /  Racing-Line.Map.Gbx  /  Splits.Map.Gbx
│
├── src/
│   ├── run_experiment.py        # Launch server / trainer / worker
│   ├── checkpoint_watcher.py    # Auto-saves timestamped checkpoints while training
│   ├── evaluate.py              # Evaluate a saved agent, no server/trainer needed
│   ├── inspect_track.py         # Print stats about a reward .pkl
│   └── envs/
│       └── step_counter.py      # Episode-stats CSV wrapper around the tmrl env
│
├── help.py                      # Built-in help for all scripts (see below)
└── requirements.txt
```

---

## Quick-start: running an experiment
 
Rename the config you want to use to `config/config.json` — that's the file `run_experiment.py` always reads:
 
```bash
# baseline (trains directly on Track 5)
cp config/config_baseline.json config/config.json
 
# OR curriculum
cp config/config_curriculum.json config/config.json
```
 
`run_experiment.py` then automatically installs it into `~/TmrlData/config/config.json` (the fixed path tmrl reads) and copies the reward `.pkl` into tmrl's reward slot before the worker starts. You never need to touch `~/TmrlData` manually.

Then open **three terminals** and run one role in each:

```bash
# terminal 1
python src/run_experiment.py --role server

# terminal 2
python src/run_experiment.py --role trainer          # add --wandb to log to W&B

# terminal 3
python src/run_experiment.py --role worker --track rewards/track5.pkl
```

Optionally, run the checkpoint watcher in a **fourth terminal** to auto-save and recover from regressions:

```bash
python src/checkpoint_watcher.py
```

Per-episode stats are written to `results/<RUN_NAME>.csv` by the worker automatically.

---

## Curriculum training

For curriculum training, repeat the three-terminal setup for each stage, swapping the track each time. An example curriculum here:

| Stage | Track file | Advance when |
|-------|-----------|--------------|
| 1 | `rewards/track1.pkl` | 10 consecutive completions or 100k steps |
| 2 | `rewards/track2.pkl` | 10 consecutive completions or 100k steps |
| 3 | `rewards/track3.pkl` | 10 consecutive completions or 100k steps |
| 4 | `rewards/track4.pkl` | 10 consecutive completions or 100k steps |
| 5 | `rewards/track5.pkl` | remaining budget |

If you wish, you can use `checkpoint_watcher.py --restore <path>` to load the best checkpoint from the previous stage before starting the next one.

---

## Evaluation

Evaluate any saved `.tmod` weights file (no server or trainer needed):

```bash
# evaluate on the training track (20 rollouts, default)
python src/evaluate.py

# evaluate a specific checkpoint on a held-out track
python src/evaluate.py \
    --weights ~/TmrlData/weights/Curriculum.tmod \
    --track   rewards/monaco_held_out.pkl \
    --runs    20 \
    --label   curriculum_monaco
```

Results are written to `results/eval_<label>.csv` and a summary row is appended to `results/eval_summary.csv`.

---

## Using the provided tracks

The `trackmania-tracks/` folder contains `.Map.Gbx` files for all tracks used in the paper. Import them into TrackMania 2020 bt copying files into **C:/Users/USER/Documents/Trackmania/Maps/My Maps** and the corresponding reward `.pkl` from `rewards/` will match.

### Recording a custom reward trajectory

To train on a track you designed yourself:
1. Load the track in TrackMania 2020.
2. Follow the tmrl [reward recording guide](https://github.com/trackmania-rl/tmrl/blob/master/readme/get_started.md) to drive a reference lap and save the `.pkl`.
3. Place the file in `rewards/` and pass it with `--track rewards/<your_file>.pkl`.

---

## Config reference

The two key fields to edit in `config/config.json` before a run:

| Field | Purpose |
|-------|---------|
| `RUN_NAME` | Name for checkpoints, W&B run, and results CSV |
| `MAX_EPOCHS` | Training budget (1 epoch ≈ 250 environment steps at default settings) |
| `WANDB_ENTITY` / `WANDB_KEY` | Required only if using `--wandb` |

All other hyperparameters (SAC learning rates, replay buffer size, episode length, failure countdown, etc.) can be tuned as you wish.

---

## Built-in help

```bash
python help.py                        # list all scripts
python help.py --script run_experiment
python help.py --script checkpoint_watcher
python help.py --script evaluate
python help.py --script inspect_track
```