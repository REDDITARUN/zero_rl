"""Training runtime using SB3 directly from generated environment code."""

from __future__ import annotations

import asyncio
import traceback
from typing import Any, Awaitable, Callable

from config import ENVS_DIR
from runtime.module_loader import load_module_from_code, resolve_env_class


ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]


def _build_model(algorithm: str, env: Any, params: dict[str, Any]) -> Any:
    from stable_baselines3 import A2C, DQN, PPO

    algorithm = algorithm.upper()
    learning_rate = float(params.get("learning_rate", 3e-4))
    gamma = float(params.get("gamma", 0.99))
    batch_size = int(params.get("batch_size", 64))
    n_steps = int(params.get("n_steps", 512))
    epsilon = float(params.get("epsilon", 0.05))

    if algorithm == "DQN":
        return DQN(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            batch_size=batch_size,
            exploration_final_eps=epsilon,
            verbose=0,
            device="cpu",
        )
    if algorithm == "A2C":
        return A2C(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            n_steps=n_steps,
            verbose=0,
            device="cpu",
        )
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size,
        verbose=0,
        device="cpu",
    )


async def run_training(
    env_id: str,
    env_code: str,
    params: dict[str, Any],
    callback: ProgressCallback,
) -> dict[str, Any]:
    """Train selected algorithm and stream progress events."""

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def emit(event: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    timesteps = int(params.get("timesteps", 5000))
    algorithm = str(params.get("algorithm", "PPO")).upper()

    async def pump(worker: asyncio.Task[dict[str, Any]]) -> dict[str, Any]:
        while True:
            if worker.done() and queue.empty():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            payload = {"env_id": env_id, **event}
            await callback(payload)

        return await worker

    def worker() -> dict[str, Any]:
        try:
            from stable_baselines3.common.callbacks import BaseCallback
            from stable_baselines3.common.monitor import Monitor

            class _TrainingProgressCallback(BaseCallback):
                """SB3 callback that pushes progress events into asyncio loop safely."""

                def __init__(self, emit: Callable[[dict[str, Any]], None]) -> None:
                    super().__init__()
                    self.emit = emit
                    self.rewards: list[float] = []
                    self.episodes = 0
                    self._last_progress_emit = 0

                def _on_step(self) -> bool:
                    if self.num_timesteps - self._last_progress_emit >= 250:
                        self._last_progress_emit = self.num_timesteps
                        self.emit(
                            {
                                "status": "running",
                                "timesteps": int(self.num_timesteps),
                                "episode": self.episodes,
                                "reward": self.rewards[-1] if self.rewards else 0.0,
                                "avg_reward_100": sum(self.rewards[-100:]) / max(len(self.rewards[-100:]), 1),
                                "message": "Training in progress",
                            }
                        )

                    infos = self.locals.get("infos") or []
                    dones = self.locals.get("dones") or []
                    for idx, done in enumerate(dones):
                        if not done or idx >= len(infos):
                            continue
                        episode = infos[idx].get("episode")
                        if episode is None:
                            continue
                        reward = float(episode.get("r", 0.0))
                        self.episodes += 1
                        self.rewards.append(reward)
                        self.emit(
                            {
                                "status": "running",
                                "timesteps": int(self.num_timesteps),
                                "episode": self.episodes,
                                "reward": reward,
                                "avg_reward_100": sum(self.rewards[-100:]) / max(len(self.rewards[-100:]), 1),
                                "message": "Episode complete",
                            }
                        )
                    return True

            module, _ = load_module_from_code(env_code, f"train_{env_id}")
            env_class = resolve_env_class(module)
            env = Monitor(env_class(render_mode=None))

            model = _build_model(algorithm, env, params)
            progress_callback = _TrainingProgressCallback(emit)
            model.learn(total_timesteps=timesteps, callback=progress_callback)

            target_dir = ENVS_DIR / env_id
            target_dir.mkdir(parents=True, exist_ok=True)
            model_path = target_dir / f"model_{algorithm.lower()}.zip"
            model.save(str(model_path))
            env.close()

            rewards = progress_callback.rewards
            return {
                "status": "complete",
                "timesteps": timesteps,
                "episode": progress_callback.episodes,
                "reward": max(rewards) if rewards else 0.0,
                "avg_reward_100": sum(rewards[-100:]) / max(len(rewards[-100:]), 1),
                "message": "Training complete",
                "payload": {
                    "model_path": str(model_path),
                    "algorithm": algorithm,
                    "episodes": progress_callback.episodes,
                    "final_avg_reward": sum(rewards[-100:]) / max(len(rewards[-100:]), 1),
                },
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "timesteps": 0,
                "episode": 0,
                "reward": 0.0,
                "avg_reward_100": 0.0,
                "message": f"Training failed: {type(exc).__name__}: {exc}",
                "payload": {"traceback": traceback.format_exc()},
            }

    await callback(
        {
            "env_id": env_id,
            "status": "running",
            "timesteps": 0,
            "episode": 0,
            "reward": 0.0,
            "avg_reward_100": 0.0,
            "message": f"Starting {algorithm} training",
        }
    )

    worker_task: asyncio.Task[dict[str, Any]] = asyncio.create_task(asyncio.to_thread(worker))
    result = await pump(worker_task)
    await callback({"env_id": env_id, **result})
    return result
