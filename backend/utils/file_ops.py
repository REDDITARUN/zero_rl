"""Filesystem utilities for ZeroRL generated environments."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from config import ENVS_DIR


def ensure_env_dir(env_id: str) -> Path:
    """Create and return environment directory."""

    path = ENVS_DIR / env_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_env_files(env_id: str, files: Dict[str, str]) -> Dict[str, str]:
    """Write generated files to disk and return absolute path mapping."""

    env_dir = ensure_env_dir(env_id)
    saved: Dict[str, str] = {}
    for name, content in files.items():
        target = env_dir / name
        target.write_text(content, encoding="utf-8")
        saved[name] = str(target)
    return saved


def save_manifest(env_id: str, name: str, prompt: str) -> None:
    """Persist environment metadata for listing and gallery endpoints."""

    env_dir = ensure_env_dir(env_id)
    manifest = {
        "env_id": env_id,
        "name": name,
        "prompt": prompt,
        "created_at": datetime.utcnow().isoformat(),
    }
    (env_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_manifest(env_dir: Path) -> Dict[str, Any]:
    """Read manifest from an environment directory."""

    manifest_path = env_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def list_envs() -> List[Dict[str, Any]]:
    """List all generated environments from disk."""

    envs: List[Dict[str, Any]] = []
    for candidate in sorted(ENVS_DIR.glob("*")):
        if not candidate.is_dir():
            continue
        data = read_manifest(candidate)
        if data:
            envs.append(data)
    return envs


def read_file(env_id: str, filename: str) -> str:
    """Read file content for a generated environment."""

    path = ENVS_DIR / env_id / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {filename}")
    return path.read_text(encoding="utf-8")
