"""Evaluation runtime for visualizing trained policy actions."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from runtime.frame_utils import encode_frame, to_jsonable
from runtime.module_loader import load_module_from_code, resolve_env_class


EvalCallback = Callable[[dict[str, Any]], Awaitable[None]]


def _load_model(algorithm: str, model_path: str) -> Any:
    from stable_baselines3 import A2C, DQN, PPO

    algo = algorithm.upper()
    if algo == "DQN":
        return DQN.load(model_path)
    if algo == "A2C":
        return A2C.load(model_path)
    return PPO.load(model_path)


async def run_evaluation(
    env_id: str,
    env_code: str,
    model_path: str,
    algorithm: str,
    episodes: int,
    max_steps: int,
    callback: EvalCallback,
) -> None:
    """Run deterministic evaluation and stream step-by-step actions."""

    module, _ = load_module_from_code(env_code, f"eval_{env_id}")
    env_class = resolve_env_class(module)
    env = env_class(render_mode="rgb_array")
    model = _load_model(algorithm, model_path)

    try:
        await callback({"env_id": env_id, "status": "running", "message": "Evaluation started"})

        for episode in range(1, episodes + 1):
            obs, info = env.reset()
            cumulative = 0.0

            for step in range(1, max_steps + 1):
                action, _ = model.predict(obs, deterministic=True)
                action_int = int(action)
                obs, reward, terminated, truncated, info = env.step(action_int)
                cumulative += float(reward)
                frame = env.render()
                if frame is None:
                    frame = env._render_frame()  # type: ignore[attr-defined]

                await callback(
                    {
                        "env_id": env_id,
                        "status": "running",
                        "episode": episode,
                        "step": step,
                        "action": action_int,
                        "reward": float(reward),
                        "cumulative_reward": cumulative,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "observation": to_jsonable(obs),
                        "info": to_jsonable(info or {}),
                        "frame": encode_frame(frame),
                    }
                )

                await asyncio.sleep(0.06)
                if terminated or truncated:
                    break

        await callback({"env_id": env_id, "status": "complete", "message": "Evaluation complete"})
    finally:
        env.close()
