"""Configuration for ZeroRL backend."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency at runtime
    load_dotenv = None


ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"

if load_dotenv is not None:
    load_dotenv(BACKEND_DIR / ".env", override=False)
    load_dotenv(ROOT_DIR / ".env", override=False)

ENVS_DIR = ROOT_DIR / "envs"
ENVS_DIR.mkdir(parents=True, exist_ok=True)

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
MAX_FIX_ATTEMPTS = int(os.getenv("MAX_FIX_ATTEMPTS", "3"))
DEFAULT_RENDER_SIZE = 512
DEFAULT_RENDER_FPS = 30

USE_CODEX_SDK = os.getenv("USE_CODEX_SDK", "true").strip().lower() in {"1", "true", "yes"}
CODEX_MODEL = os.getenv("CODEX_MODEL", "gpt-5-codex")
CODEX_MODEL_ARCHITECT = os.getenv("CODEX_MODEL_ARCHITECT", CODEX_MODEL)
CODEX_MODEL_REWARDS = os.getenv("CODEX_MODEL_REWARDS", CODEX_MODEL)
CODEX_MODEL_SPACES = os.getenv("CODEX_MODEL_SPACES", CODEX_MODEL)
CODEX_MODEL_VALIDATOR = os.getenv("CODEX_MODEL_VALIDATOR", CODEX_MODEL)
CODEX_MODEL_DOCS = os.getenv("CODEX_MODEL_DOCS", CODEX_MODEL)
CODEX_MODEL_TRAINER = os.getenv("CODEX_MODEL_TRAINER", CODEX_MODEL)
CODEX_TIMEOUT_SEC = int(os.getenv("CODEX_TIMEOUT_SEC", "480"))
CODEX_MAX_RETRIES = int(os.getenv("CODEX_MAX_RETRIES", "2"))
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
