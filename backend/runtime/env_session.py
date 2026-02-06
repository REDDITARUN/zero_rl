"""Interactive environment runtime manager for reset/step/state APIs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from runtime.frame_utils import encode_frame, to_jsonable
from runtime.module_loader import load_module_from_code, resolve_env_class

MAX_UI_ACTIONS = 64


@dataclass
class EnvSession:
    """Live environment session state."""

    code_hash: str
    env: Any
    action_labels: list[str]
    step_count: int = 0
    cumulative_reward: float = 0.0
    last_reward: float = 0.0
    last_action: str | None = None
    terminated: bool = False
    truncated: bool = False
    last_obs: Any = None
    last_info: dict[str, Any] = field(default_factory=dict)
    last_frame: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)


class EnvRuntimeManager:
    """Tracks active env instances for user-controlled stepping."""

    def __init__(self) -> None:
        self.sessions: dict[str, EnvSession] = {}

    def reset(self, env_id: str, env_code: str, action_labels: list[str]) -> dict[str, Any]:
        """Reset session and return current runtime state."""

        session = self._get_or_create(env_id, env_code, action_labels)
        try:
            obs, info = session.env.reset()
            frame = session.env.render()
            if frame is None:
                frame = session.env._render_frame()  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            if not self._should_recreate(exc):
                raise
            session = self._get_or_create(env_id, env_code, action_labels, force_recreate=True)
            obs, info = session.env.reset()
            frame = session.env.render()
            if frame is None:
                frame = session.env._render_frame()  # type: ignore[attr-defined]

        session.step_count = 0
        session.cumulative_reward = 0.0
        session.last_reward = 0.0
        session.last_action = None
        session.terminated = False
        session.truncated = False
        session.last_obs = to_jsonable(obs)
        session.last_info = to_jsonable(info or {})
        session.last_frame = encode_frame(frame)
        session.history = []

        return self._to_payload(env_id, session)

    def step(self, env_id: str, env_code: str, action_labels: list[str], action: int) -> dict[str, Any]:
        """Apply one action and return updated runtime state."""

        session = self._get_or_create(env_id, env_code, action_labels)

        if session.terminated or session.truncated:
            return self._to_payload(env_id, session)

        try:
            obs, reward, terminated, truncated, info = session.env.step(int(action))
            frame = session.env.render()
            if frame is None:
                frame = session.env._render_frame()  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            if not self._should_recreate(exc):
                raise
            session = self._get_or_create(env_id, env_code, action_labels, force_recreate=True)
            obs, info = session.env.reset()
            frame = session.env.render()
            if frame is None:
                frame = session.env._render_frame()  # type: ignore[attr-defined]
            reward = 0.0
            terminated = False
            truncated = False

        session.step_count += 1
        session.last_reward = float(reward)
        session.cumulative_reward += float(reward)
        session.terminated = bool(terminated)
        session.truncated = bool(truncated)
        session.last_obs = to_jsonable(obs)
        session.last_info = to_jsonable(info or {})
        session.last_frame = encode_frame(frame)

        action_label = action_labels[action] if 0 <= action < len(action_labels) else str(action)
        session.last_action = action_label
        session.history.append(
            {
                "step": session.step_count,
                "action": action_label,
                "reward": session.last_reward,
                "cumulative_reward": session.cumulative_reward,
                "terminated": session.terminated,
                "truncated": session.truncated,
            }
        )
        if len(session.history) > 80:
            session.history = session.history[-80:]

        return self._to_payload(env_id, session)

    def get_state(self, env_id: str, env_code: str, action_labels: list[str]) -> dict[str, Any]:
        """Return current runtime state; create/reset if missing."""

        if env_id not in self.sessions:
            return self.reset(env_id, env_code, action_labels)
        session = self._get_or_create(env_id, env_code, action_labels)
        return self._to_payload(env_id, session)

    def get_action_labels(self, env_id: str, env_code: str, action_labels: list[str]) -> list[str]:
        """Return runtime-validated action labels for control mapping."""

        session = self._get_or_create(env_id, env_code, action_labels)
        return list(session.action_labels)

    def _get_or_create(
        self,
        env_id: str,
        env_code: str,
        action_labels: list[str],
        force_recreate: bool = False,
    ) -> EnvSession:
        code_hash = hashlib.sha256(env_code.encode("utf-8")).hexdigest()
        current = self.sessions.get(env_id)
        if not force_recreate and current is not None and current.code_hash == code_hash:
            return current

        if current is not None:
            try:
                current.env.close()
            except Exception:
                pass

        module, _ = load_module_from_code(env_code, env_id)
        env_class = resolve_env_class(module)
        env = env_class(render_mode="rgb_array")

        normalized = [str(label) for label in action_labels]
        model_labels = getattr(env, "action_labels", None)
        if isinstance(model_labels, list):
            candidate = [str(label) for label in model_labels if str(label).strip()]
            if candidate:
                normalized = candidate

        n = getattr(getattr(env, "action_space", None), "n", None)
        if isinstance(n, int) and n > 0:
            if len(normalized) != n:
                normalized = [f"action_{idx}" for idx in range(n)]
        if not normalized:
            normalized = ["action_0", "action_1", "action_2", "action_3"]
        if len(normalized) > MAX_UI_ACTIONS:
            normalized = normalized[:MAX_UI_ACTIONS]

        session = EnvSession(code_hash=code_hash, env=env, action_labels=normalized)
        self.sessions[env_id] = session
        return session

    def _should_recreate(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "library not initialized" in message or "video system not initialized" in message

    def _to_payload(self, env_id: str, session: EnvSession) -> dict[str, Any]:
        return {
            "env_id": env_id,
            "step": session.step_count,
            "last_action": session.last_action,
            "last_reward": session.last_reward,
            "cumulative_reward": session.cumulative_reward,
            "terminated": session.terminated,
            "truncated": session.truncated,
            "observation": session.last_obs,
            "info": session.last_info,
            "frame": session.last_frame,
            "available_actions": session.action_labels,
            "history": session.history,
        }


env_runtime_manager = EnvRuntimeManager()
