"""Execution helpers for environment generation runtime."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from config import ENVS_DIR


def zip_environment(env_id: str) -> bytes:
    """Create zip archive bytes for a generated environment folder."""

    env_dir = ENVS_DIR / env_id
    if not env_dir.exists():
        raise FileNotFoundError(f"Environment not found: {env_id}")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in env_dir.glob("**/*"):
            if path.is_file():
                archive.write(path, arcname=str(Path(env_id) / path.relative_to(env_dir)))
    return buffer.getvalue()
