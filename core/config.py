"""Project-wide configuration: paths, model selection, .env loading.

Per-tool LLM configuration via env vars:

  # Global defaults (fallback for all tools)
  LLM_PROVIDER=anthropic
  LLM_MODEL=claude-sonnet-4-6
  LLM_TEMPERATURE=0.2

  # Per-tool overrides (optional — falls back to global)
  CAD_PROVIDER=google-genai
  CAD_MODEL=gemini-2.0-flash
  CAD_TEMPERATURE=0.2

  URDF_PROVIDER=anthropic
  URDF_MODEL=claude-sonnet-4-6
  URDF_TEMPERATURE=0.2

  ORCHESTRATOR_PROVIDER=anthropic
  ORCHESTRATOR_MODEL=claude-sonnet-4-6
  ORCHESTRATOR_TEMPERATURE=0.2
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
ENVS_DIR = ROOT_DIR / "envs"
ASSETS_DIR = ROOT_DIR / "assets"
SKILLS_DIR = ROOT_DIR / "core" / "prompts" / "skills"
DOCS_DIR = ROOT_DIR / "docs"

ENVS_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)

# ── LLM defaults ─────────────────────────────────────────────────────────────

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# Provider → env var that holds the API key
_KEY_ENV_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google-genai": "GOOGLE_API_KEY",
}


def get_api_key(provider: str | None = None) -> str:
    """Return the API key for the given (or default) provider."""
    provider = provider or DEFAULT_PROVIDER
    env_var = _KEY_ENV_VARS.get(provider, f"{provider.upper()}_API_KEY")
    key = os.getenv(env_var, "")
    if not key:
        raise EnvironmentError(
            f"Missing {env_var}. Set it in .env or your shell environment."
        )
    return key


# ── Per-tool LLM config ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolLLMConfig:
    """LLM configuration for a specific tool."""

    provider: str
    model: str
    temperature: float

    @property
    def api_key(self) -> str:
        return get_api_key(self.provider)

    @property
    def vlm_model_name(self) -> str:
        """Model name compatible with articulate_anything's setup_vlm_model.

        The VLM factory dispatches on substring: 'claude', 'gpt', 'gemini'.
        Ensure the model name contains the provider keyword.
        """
        m = self.model
        if self.provider == "anthropic" and "claude" not in m:
            m = f"claude-{m}"
        elif self.provider == "openai" and "gpt" not in m:
            m = f"gpt-{m}"
        elif self.provider == "google-genai" and "gemini" not in m:
            m = f"gemini-{m}"
        return m

    def __repr__(self) -> str:
        return f"{self.provider}/{self.model} (t={self.temperature})"


def _tool_config(prefix: str) -> ToolLLMConfig:
    """Build a ToolLLMConfig from env vars with the given prefix.

    Example: prefix="CAD" reads CAD_PROVIDER, CAD_MODEL, CAD_TEMPERATURE,
    falling back to LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE.
    """
    return ToolLLMConfig(
        provider=os.getenv(f"{prefix}_PROVIDER", DEFAULT_PROVIDER),
        model=os.getenv(f"{prefix}_MODEL", DEFAULT_MODEL),
        temperature=float(os.getenv(f"{prefix}_TEMPERATURE", str(DEFAULT_TEMPERATURE))),
    )


# Pre-built configs for each tool — import these in tool files
CAD_CONFIG = _tool_config("CAD")
URDF_CONFIG = _tool_config("URDF")
ORCHESTRATOR_CONFIG = _tool_config("ORCHESTRATOR")


# ── LangChain chat model (for orchestrator) ──────────────────────────────────

def get_chat_model(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    **kwargs: Any,
):
    """Instantiate a LangChain chat model for the chosen provider."""
    from langchain.chat_models import init_chat_model

    provider = provider or ORCHESTRATOR_CONFIG.provider
    model = model or ORCHESTRATOR_CONFIG.model
    temperature = temperature if temperature is not None else ORCHESTRATOR_CONFIG.temperature

    return init_chat_model(
        model=model,
        model_provider=provider,
        temperature=temperature,
        api_key=get_api_key(provider),
        **kwargs,
    )


# ── Validation defaults ──────────────────────────────────────────────────────

MAX_FIX_ATTEMPTS = int(os.getenv("MAX_FIX_ATTEMPTS", "3"))
VALIDATION_ROLLOUT_STEPS = int(os.getenv("VALIDATION_ROLLOUT_STEPS", "10"))
