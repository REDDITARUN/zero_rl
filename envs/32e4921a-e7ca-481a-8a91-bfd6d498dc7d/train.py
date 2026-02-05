"""Training script for ZeroMazeEnv."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from env import ZeroMazeEnv


class ProgressCallback(BaseCallback):
    """Streams JSON progress records for websocket relays."""

    def __init__(self) -> None:
        super().__init__()
        self.rewards: list[float] = []
        self.episodes = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for idx, done in enumerate(dones):
            if done and idx < len(infos) and "episode" in infos[idx]:
                episode = infos[idx]["episode"]
                reward = float(episode.get("r", 0.0))
                self.episodes += 1
                self.rewards.append(reward)
                payload: Dict[str, Any] = {
                    "type": "training_progress",
                    "episode": self.episodes,
                    "reward": reward,
                    "avg_reward_100": float(np.mean(self.rewards[-100:])),
                    "timesteps": self.num_timesteps,
                }
                print(json.dumps(payload), flush=True)
        return True


def train(total_timesteps: int = 10000) -> None:
    env = Monitor(ZeroMazeEnv())
    check_env(env)

    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=10, verbose=0, device="cpu")
    model.learn(total_timesteps=total_timesteps, callback=ProgressCallback())
    model.save("model.zip")
    print(json.dumps({"type": "complete", "timesteps": total_timesteps}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10000)
    args = parser.parse_args()
    train(total_timesteps=args.timesteps)
