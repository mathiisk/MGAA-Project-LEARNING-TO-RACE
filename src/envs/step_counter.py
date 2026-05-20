import csv
import time
from pathlib import Path


class StepCounterWrapper:
    """
    Wraps a TMRL gym env to log episode results to a CSV.

    Appends to an existing CSV rather than overwriting it, so restarting
    the worker with the same RUN_NAME never loses previous run data.
    Each new worker session is marked with a separator row so you can
    tell runs apart in the CSV and in plots.

    CSV columns:
        wall_time      — seconds since this session started
        env_steps      — cumulative steps across all sessions
        episode        — cumulative episode number across all sessions
        episode_reward — total reward for this episode
        episode_length — number of steps in this episode
    """

    def __init__(self, env, run_name: str, results_dir: str = "results"):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        self._csv_path = results_path / f"{run_name}.csv"

        # Read existing data so we can continue episode/step counts
        self._episode, self._env_steps = self._read_last_counts()

        file_existed = self._csv_path.exists() and self._csv_path.stat().st_size > 0

        # Always append — never overwrite
        self._csv_file = open(self._csv_path, "a", newline="")
        self._writer = csv.writer(self._csv_file)

        if not file_existed:
            # Fresh file — write header
            self._writer.writerow(["wall_time", "env_steps", "episode",
                                   "episode_reward", "episode_length"])
        else:
            # Existing file — write a separator so runs are distinguishable
            import datetime
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._writer.writerow([])
            self._writer.writerow([f"# --- new run started {ts} "
                                   f"(continuing from episode {self._episode}, "
                                   f"step {self._env_steps}) ---"])
            self._writer.writerow([])

        self._csv_file.flush()

        self._episode_reward = 0.0
        self._episode_length = 0
        self._start_time = time.time()

        print(f"[StepCounterWrapper] logging to {self._csv_path}")
        if file_existed:
            print(f"[StepCounterWrapper] continuing from episode {self._episode}, "
                  f"step {self._env_steps}")

    def _read_last_counts(self) -> tuple[int, int]:
        """
        Read the last episode number and env_steps from an existing CSV.
        Returns (0, 0) if the file doesn't exist or has no data rows.
        """
        if not self._csv_path.exists():
            return 0, 0
        try:
            last_episode = 0
            last_steps = 0
            with open(self._csv_path, "r", newline="") as f:
                for line in f:
                    line = line.strip()
                    # Skip blank lines and comment/separator rows
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(",")
                    if len(parts) < 3:
                        continue
                    try:
                        ep = int(parts[2])
                        steps = int(parts[1])
                        last_episode = max(last_episode, ep)
                        last_steps = max(last_steps, steps)
                    except ValueError:
                        # header row or malformed — skip
                        continue
            return last_episode, last_steps
        except Exception:
            return 0, 0

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._episode_reward = 0.0
        self._episode_length = 0
        return result

    def step(self, action):
        result = self.env.step(action)

        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
            terminated, truncated = done, False

        self._env_steps += 1
        self._episode_reward += float(reward)
        self._episode_length += 1

        if done:
            self._episode += 1
            self._writer.writerow([
                round(time.time() - self._start_time, 2),
                self._env_steps,
                self._episode,
                round(self._episode_reward, 4),
                self._episode_length,
            ])
            self._csv_file.flush()  # write immediately so data survives a crash

        if len(result) == 5:
            return obs, reward, terminated, truncated, info
        return obs, reward, done, info

    def close(self):
        self._csv_file.close()
        self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)