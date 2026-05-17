import argparse
import pickle

try:
    import gymnasium as gym
except ModuleNotFoundError:
    try:
        import gym
    except ModuleNotFoundError:
        gym = None
        
import numpy as np

# for racing line smoothing
from scipy.ndimage import uniform_filter1d


# some helper functions below 
#----------------------------#

def load_centerline(pkl_path: str) -> np.ndarray:
    """
    Load in the reward okl file (N, 3) float64 array of [x, y, z] track positions.
    This acts as a center line.
    """
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
        
    # assert file is correct 
    assert isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 3, (
        f"Expected (N, 3) ndarray, got {type(data)} shape={getattr(data, 'shape', None)}"
    )
    
    return data.astype(np.float64)


def smooth_racing_line(centerline: np.ndarray, window: int = 50) -> np.ndarray:
    """
    Smooth the centerline with a moving average
    Idea is approximate apex-cutting. On straight sections the smoothed line equals the centerline.
    On corners, it cuts the inside slightly.
    A larger "window" equals more agressive smoothing / corner cutting.
    

    Args:
        centerline (np.ndarray): Loaded centerline ndarray from pkl file
        window (int, optional): How much smoothing . Defaults to 50.

    Returns:
        np.ndarray: Smoothed centerline ndarray
    """
    smoothed = np.stack([uniform_filter1d(centerline[:, i], size = window, mode="wrap") for i in range (3)], axis=1)
    return smoothed


def find_closest_index(car_pos_xz: np.ndarray, line_xz: np.ndarray) -> int:
    """
    Return index of waypoints on "line_xz" closest to "car_pos_xz"

    Args: #TODO
        car_pos_xz (np.ndarray): _description_
        line_xz (np.ndarray): _description_

    Returns: #TODO
        int: _description_
    """
    diffs = line_xz - car_pos_xz 
    dists = np.einsum("ij,ij->i", diffs, diffs)
    return int(np.argmin(dists))


def get_lookahead_waypoints(car_xz: np.ndarray, line: np.ndarray, n_waypoints: int = 5, stride: int = 10) -> np.ndarray:
    """Returns "n_waypoints" futute waypoints as offsets relative to the car.

    Args:
        car_xz (np.ndarray): (2,) car position in [x, z]
        line (np.ndarray): (N, 3) centerline or racing line
        n_waypoints (int, optional): How many future points to include in observation. Defaults to 5.
        stride (int, optional): Spacing between sampled waypoints (in raw-point units). Defaults to 10.
                                At 0.1m/point we get 1m between points

    Returns: 
        np.ndarray: (n_waypoints * 2,) array of (dx, dz) offsets, normalised by a fixed scale so values stay reasonable
    """
    line_xz = [line[:, 0, 2]] # drop y, thus (N, 2)
    closest = find_closest_index(car_xz, line_xz)
    n_total = len(line_xz)
    
    waypoints = []
    for i in range(1, n_waypoints + 1):
        idx = (closest + i * stride) % n_total
        offset = line_xz[idx] - car_xz # relative pos
        waypoints.append(offset)
        
    flat = np.concatenate(waypoints)
    flat = flat / 10.0 # sort of normalization
    return flat.astype(np.float32)



# gym wrapper below :)
#---------------------#

_GymWrapper = gym.Wrapper if gym is not None else object

class WaypointAugmentedEnv(_GymWrapper):
    def __init__(self, env, track_pkl: str, variant: str = "C", n_waypoints: int = 5, stride: int = 10, smooth_window: int = 50):
        if gym is not None:
            super().__init__(env)
        else:
            self.env = env
            self.action_space = env.action_space
        assert variant in ("C", "D"), "variant must be 'C' or 'D'"
        self.variant = variant
        self.n_waypoints = n_waypoints
        self.stride = stride
        
        
        centerline = load_centerline(track_pkl)
        if variant == "D":
            self.line = smooth_racing_line(centerline, window=smooth_window)
        else:
            self.line = centerline
            
        
        # extend observation space
        base_obs_space = env.observation_space
        extra = n_waypoints * 2
        new_low = np.concatenate([base_obs_space.low, np.full(extra, -np.inf, dtype=np.float32)])
        new_high = np.concatenate([base_obs_space.high, np.full(extra, np.inf, dtype=np.float32)])
        
        # build minimal box-like space
        if gym is not None:
            self.observation_space = gym.spaces.Box(low=new_low, high=new_high, dtype=np.float32)
        else:
            class _Box:
                def __init__(self, low, high):
                    self.low, self.high = low, high
                    self.shape = low.shape
                    self.dtype = np.float32
            self.observation_space = _Box(new_low, new_high)
        self._extra_dim = extra
        
    
        """
        Extract car (x, z) from TMRL observations
        TMRL LiDAR obs is flat vector with layour as:
            [0] speed
            [1..4] previous action
            [5..80] lidar beams [19 beams x 4 frames history]
            
        Important: TMRL does NOT give us abosulte car position in the observations
        But we have LiDAR + speed, thus we retrieve position via env.unwrapped.
        If not available, we simply fall back to closest waypoint from last step.
        """
        
    def _get_car_xz(self) -> np.ndarray:
        """
        Attpemt to get (x, z) from the TMRL env.

        Returns: #TODO
            np.ndarray: _description_
        """
        try:
            state = self.env.unwrapped.interface.game_state # TMRL stores last game state here
            x, z = float(state[3]), float(state[5]) # game_state is [speed, gear, rpm, x, y, z, ...]
            return np.array([x, z], dtype=np.float64)
        except Exception:
            # not great, but prevents crash, uses centroid of whole track
            return self.line[0, [0,2]].copy()
            

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        """#TODO
        _summary_ 

        Args: #TODO
            obs (np.ndarray): _description_

        Returns: #TODO
            np.ndarray: _description_
        """
        car_xz = self._get_car_xz()
        extra = get_lookahead_waypoints(car_xz, self.line, self.n_waypoints, self.stride)
        return np.concatenate([obs.flatten(), extra]).astype(np.float32)
        
        
    def reset(self, **kwargs):
        """#TODO
        _summary_

        Returns:#TODO
            _type_: _description_
        """
        result = self.env.reset(**kwargs)
        # depending on gym version, reset() can return (obs, info) or just obs
        if isinstance(result, tuple):
            obs, info = result
            return self._augment(obs), info
        return self._augment(result)
    
    
    def step(self, action):
        """#TODO
        _summary_

        Args:#TODO
            action (_type_): _description_

        Returns:#TODO
            _type_: _description_
        """
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return self._augment(obs), reward, terminated, truncated, info
        obs, reward, done, info = result
        return self._augment(obs), reward, done, info
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["C", "D"], default="C")
    parser.add_argument("--track", default="rewards/reward_track1.pkl")
    parser.add_argument("--n_waypoints", type=int, default=5)
    args = parser.parse_args()
    
        
        