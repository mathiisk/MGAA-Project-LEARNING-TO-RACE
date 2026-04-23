import sys
import numpy as np
import torch
import torch.nn as nn

from tmrl.util import partial
from tmrl.networking import Trainer, RolloutWorker, Server
from tmrl.training_offline import TrainingOffline, TorchTrainingOffline
from tmrl.actor import TorchActorModule
from tmrl.training import TrainingAgent

import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj


epochs = cfg.TMRL_CONFIG["MAX_EPOCHS"]
rounds = cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"]
steps = cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"]
update_buffer_interval = cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"]
update_model_interval = cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"]
max_samples_per_episode = cfg.TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"]

memory_base = cfg_obj.MEM
sample_compressor = cfg_obj.SAMPLE_COMPRESSOR
obs_preprocessor = cfg_obj.OBS_PREPROCESSOR

# PPO-specific hyperparameters (tune these)
CLIP_RATIO = 0.2
PPO_EPOCHS = 10
GAMMA = 0.99
LAM = 0.95
LR = 3e-4



class PPOActor(TorchActorModule):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        # LIDAR obs: 19 rays + speed + gear + rpm = ~22 inputs
        self.net = nn.Sequential(
            nn.Linear(22, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.mean = nn.Linear(64, action_space.shape[0])
        self.log_std = nn.Linear(64, action_space.shape[0])

    def forward(self, obs):
        # TODO: verify obs shape matches LIDAR interface output
        x = self.net(obs)
        return self.mean(x), self.log_std(x).clamp(-2, 2)

    def act(self, obs, test=False):
        mean, log_std = self.forward(obs)
        if test:
            return torch.tanh(mean)
        dist = torch.distributions.Normal(mean, log_std.exp())
        return torch.tanh(dist.sample())


# TRAINER (PPO update logic) — not yet implemented
class PPOTrainer(TrainingAgent):
    # TODO: implement PPO
    # Steps:
    # 1. Store rollout buffer (obs, actions, rewards, dones, log_probs, values)
    # 2. Compute advantages using GAE (gamma, lambda)
    # 3. For PPO_EPOCHS iterations:
    #    a. Compute new log_probs and values
    #    b. Compute ratio = exp(new_log_probs - old_log_probs)
    #    c. Clipped surrogate loss: min(ratio * adv, clip(ratio, 1±CLIP_RATIO) * adv)
    #    d. Value loss: MSE(values, returns)
    #    e. Backprop and update

    def __init__(self, observation_space, action_space, device=None):
        super().__init__(observation_space, action_space, device)
        raise NotImplementedError("PPOTrainer not yet implemented")

    def train(self, batch):
        raise NotImplementedError("PPOTrainer not yet implemented")

    def get_actor(self):
        raise NotImplementedError("PPOTrainer not yet implemented")


def run_server():
    server = Server()
    while True:  # server runs indefinitely
        server.run()


def run_trainer():
    training_agent = partial(PPOTrainer)
    training_offline = TorchTrainingOffline(
        env_cls=cfg_obj.ENV_CLS,
        memory_cls=memory_base,
        training_agent_cls=training_agent,
        epochs=epochs,
        rounds=rounds,
        steps=steps,
        update_buffer_interval=update_buffer_interval,
        update_model_interval=update_model_interval,
        max_training_steps_per_env_step=1.0,
    )
    trainer = Trainer(
        training_cls=training_offline,
        server_ip=cfg.SERVER_IP_FOR_TRAINER,
        model_path=cfg.MODEL_PATH_TRAINER,
        checkpoint_path=cfg.CHECKPOINT_PATH,
        dump_run_instance_fn=None,
        load_run_instance_fn=None,
    )
    trainer.run()


def run_worker():
    worker = RolloutWorker(
        env_cls=cfg_obj.ENV_CLS,
        actor_module_cls=partial(PPOActor),
        sample_compressor=sample_compressor,
        device="cpu",
        server_ip=cfg.SERVER_IP_FOR_WORKER,
        max_samples_per_episode=max_samples_per_episode,
        obs_preprocessor=obs_preprocessor,
        model_path=cfg.MODEL_PATH_WORKER,
        crc_debug=False,
        model_path_update_buffer=cfg.REPLAY_MEMORY_PATH,
    )
    worker.run()


def run_test():
    worker = RolloutWorker(
        env_cls=cfg_obj.ENV_CLS,
        actor_module_cls=partial(PPOActor),
        sample_compressor=sample_compressor,
        device="cpu",
        server_ip=cfg.SERVER_IP_FOR_WORKER,
        max_samples_per_episode=max_samples_per_episode,
        obs_preprocessor=obs_preprocessor,
        model_path=cfg.MODEL_PATH_WORKER,
        crc_debug=False,
        model_path_update_buffer=cfg.REPLAY_MEMORY_PATH,
    )
    worker.run_episodes(10000, train=False)


if __name__ == "__main__":
    if "--server" in sys.argv:
        run_server()
    elif "--trainer" in sys.argv:
        run_trainer()
    elif "--worker" in sys.argv:
        run_worker()
    elif "--test" in sys.argv:
        run_test()
    else:
        print("Usage: python pipeline.py [--server | --trainer | --worker | --test]")