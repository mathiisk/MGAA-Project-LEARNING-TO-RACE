import csv
import time
from pathlib import Path


class StepCounterWrapper:
    """_summary_ #TODO
    """

    def __init__(self, env, run_name: str, results_dir: str = "results"):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        self._csv_path = results_path / f"{run_name}.csv"

        self._csv_file = open(self._csv_path, "w", newline="")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow(["wall_time", "env_steps", "episode", "episode_reward", "episode_length"])
        self._csv_file.flush()

        self._env_steps = 0
        self._episode = 0
        self._episode_reward = 0.0
        self._episode_length = 0
        self._start_time = time.time()

        print(f"[StepCounterWrapper] logging to {self._csv_path}")

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
            self._csv_file.flush()   # write immediately so data survives a crash

        if len(result) == 5:
            return obs, reward, terminated, truncated, info
        return obs, reward, done, info

    def close(self):
        self._csv_file.close()
        self.env.close()

    def __getattr__(self, name):
        # delegate anything else (render, seed, etc.) to the inner env
        return getattr(self.env, name)
