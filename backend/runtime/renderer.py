"""Rendering pipeline for environment rgb frames."""

from __future__ import annotations

from runtime.frame_utils import encode_frame
from runtime.module_loader import load_module_from_code, resolve_env_class


def render_frame_base64_from_code(env_id: str, env_code: str) -> str:
    """Load generated env code and return a PNG frame as base64 string."""

    module, _ = load_module_from_code(env_code, f"render_{env_id}")
    env_cls = resolve_env_class(module)

    env = env_cls(render_mode="rgb_array")
    try:
        env.reset()
        frame = env.render()
        if frame is None:
            frame = env._render_frame()  # type: ignore[attr-defined]
        return encode_frame(frame)
    finally:
        env.close()
