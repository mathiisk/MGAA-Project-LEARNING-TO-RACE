"""
Each variant has its own config file in config/. The launcher sets
TMRL_CONFIG_PATH to point at the right one before TMRL loads anything,
so every variant gets isolated model weights, checkpoints, and replay memory.

Usage (open 3 terminals, same command structure for every variant):
    python src/run_experiment.py --variant A --role server
    python src/run_experiment.py --variant A --role trainer
    python src/run_experiment.py --variant A --role worker

    python src/run_experiment.py --variant C --role server --track rewards/reward_track1.pkl
    python src/run_experiment.py --variant C --role trainer --track rewards/reward_track1.pkl
    python src/run_experiment.py --variant C --role worker  --track rewards/reward_track1.pkl

Variants
--------
    A  —  raw LiDAR only       (81-dim obs)
    B  —  image / TM20FULL     (image stacked obs)
    C  —  LiDAR + waypoints    (91-dim obs, centerline lookahead)
    D  —  LiDAR + racing line  (91-dim obs, smoothed centerline lookahead)

Config files (config/ folder)
------------------------------
    config/config_variantA.json   RUN_NAME=SAC_LIDAR_variantA   TM20LIDAR
    config/config_variantB.json   RUN_NAME=SAC_IMAGE_variantB   TM20FULL
    config/config_variantC.json   RUN_NAME=SAC_LIDAR_variantC   TM20LIDAR
    config/config_variantD.json   RUN_NAME=SAC_LIDAR_variantD   TM20LIDAR

Files written per variant (inside %TMRL_DATA%):
    weights/SAC_LIDAR_variantC_trainer.pth
    weights/SAC_LIDAR_variantC_worker.pth
    checkpoints/SAC_LIDAR_variantC.pkl
    replay_memory/SAC_LIDAR_variantC/

Offline sanity check (no TrackMania needed):
    python src/envs/augmented_lidar_env.py --variant C --track rewards/reward_track1.pkl
    python src/envs/augmented_lidar_env.py --variant D --track rewards/reward_track2.pkl
"""

import argparse
import os
import sys
from pathlib import Path

# Root of the project (one level above src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR   = PROJECT_ROOT / "config"


def _set_config(variant: str):
    """
    Point TMRL_CONFIG_PATH at the right config file before any tmrl import.
    TMRL reads this env var on first import of config_constants.
    Must be called before any `import tmrl` statement.
    """
    config_path = CONFIG_DIR / f"config_variant{variant}.json"
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}")
        print(f"  Expected files: {CONFIG_DIR}/config_variant{{A,B,C,D}}.json")
        sys.exit(1)
    os.environ["TMRL_CONFIG_PATH"] = str(config_path)
    print(f"[run_experiment] config -> {config_path}")



def make_augmented_env_cls(track_pkl: str, variant: str, n_waypoints: int, stride: int):
    """
    Return an env *class* (no-arg callable) that TMRL can instantiate.
    TMRL calls env_cls() internally, so we wrap with a closure.
    """
    import tmrl.config.config_objects as cfg_obj
    from envs.augmented_lidar_env import WaypointAugmentedEnv

    base_cls = cfg_obj.ENV_CLS

    class AugmentedEnvCls:
        def __new__(cls):
            base_env = base_cls()
            return WaypointAugmentedEnv(
                base_env,
                track_pkl=track_pkl,
                variant=variant,
                n_waypoints=n_waypoints,
                stride=stride,
            )

    return AugmentedEnvCls


def run_server():
    import tmrl.config.config_constants as cfg
    from tmrl.networking import Server

    server = Server(
        port=cfg.PORT,
        header_size=cfg.HEADER_SIZE,
        buffer_size=cfg.BUFFER_SIZE,
    )
    while True:
        server.run()


def run_trainer(env_cls):
    import tmrl.config.config_constants as cfg
    import tmrl.config.config_objects as cfg_obj
    from tmrl.networking import Trainer
    from tmrl.training_offline import TorchTrainingOffline

    training = TorchTrainingOffline(
        env_cls=env_cls,
        memory_cls=cfg_obj.MEM,
        training_agent_cls=cfg_obj.TRAINER,
        epochs=cfg.TMRL_CONFIG["MAX_EPOCHS"],
        rounds=cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"],
        steps=cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"],
        update_buffer_interval=cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"],
        update_model_interval=cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"],
        max_training_steps_per_env_step=cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"],
    )
    trainer = Trainer(
        training_cls=training,
        server_ip=cfg.SERVER_IP_FOR_TRAINER,
        model_path=cfg.MODEL_PATH_TRAINER,
        checkpoint_path=cfg.CHECKPOINT_PATH,
    )
    trainer.run()


def run_worker(env_cls):
    import tmrl.config.config_constants as cfg
    import tmrl.config.config_objects as cfg_obj
    from tmrl.networking import RolloutWorker
    from tmrl.util import partial

    worker = RolloutWorker(
        env_cls=env_cls,
        actor_module_cls=partial(cfg_obj.POLICY),
        sample_compressor=cfg_obj.SAMPLE_COMPRESSOR,
        device="cpu",
        server_ip=cfg.SERVER_IP_FOR_WORKER,
        max_samples_per_episode=cfg.TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"],
        obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
        model_path=cfg.MODEL_PATH_WORKER,
        crc_debug=False,
        model_path_update_buffer=cfg.REPLAY_MEMORY_PATH,
    )
    worker.run()
    
    
def main():
    parser = argparse.ArgumentParser(
        description="Unified launcher for all observation-space ablation variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--variant", choices=["A", "B", "C", "D"], required=True,
        help="A=LiDAR only, B=image, C=LiDAR+waypoints, D=LiDAR+racing-line",
    )
    parser.add_argument(
        "--role", choices=["server", "trainer", "worker"], required=True,
        help="Which TMRL process to launch",
    )
    parser.add_argument(
        "--track", default="rewards/reward_track1.pkl",
        help="Path to reward .pkl (centerline). Used for variants C and D only.",
    )
    parser.add_argument(
        "--n_waypoints", type=int, default=5,
        help="Number of lookahead waypoints added to obs (C/D only). Default: 5",
    )
    parser.add_argument(
        "--stride", type=int, default=10,
        help="Waypoint stride in raw-point units (10 -> 1 m apart). Default: 10",
    )
    args = parser.parse_args()

    # Set config path BEFORE any tmrl import
    _set_config(args.variant)

    # For A and B: use TMRL's built-in env (no wrapper needed)
    # For C and D: wrap with WaypointAugmentedEnv
    if args.variant in ("A", "B"):
        import tmrl.config.config_objects as cfg_obj
        env_cls = cfg_obj.ENV_CLS
        obs_note = "81-dim LiDAR" if args.variant == "A" else "image stack (TM20FULL)"
    else:
        env_cls = make_augmented_env_cls(
            track_pkl=args.track,
            variant=args.variant,
            n_waypoints=args.n_waypoints,
            stride=args.stride,
        )
        obs_note = f"{81 + args.n_waypoints * 2}-dim LiDAR+waypoints"

    print(f"[run_experiment] variant={args.variant} | role={args.role} | obs={obs_note}\n")

    if args.role == "server":
        run_server()
    elif args.role == "trainer":
        run_trainer(env_cls)
    elif args.role == "worker":
        run_worker(env_cls)


if __name__ == "__main__":
    main()
