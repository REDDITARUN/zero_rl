"""Deterministic validation pipeline for generated Gymnasium environments.

Stages (per ARCHITECTURE.md Section 7):
  1. file_check  — env.py exists
  2. syntax      — py_compile
  3. import      — importlib
  4. instantiate — env = MyEnv()
  5. reset       — obs, info = env.reset(); check obs shape
  6. step        — N random steps, check rewards/dones/obs
  7. render      — render one RGB frame
  8. check_env   — stable_baselines3.check_env()

Genesis envs are validated in a subprocess to avoid macOS OpenGL/AppKit
thread-safety crashes (NSInternalInconsistencyException).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.tools import tool

from core.config import ENVS_DIR, VALIDATION_ROLLOUT_STEPS

_VALIDATE_SCRIPT = Path(__file__).parent / "validate_subprocess.py"
_GENESIS_SUBPROCESS_TIMEOUT = 300  # 5 minutes for Genesis builds


@dataclass
class ValidationResult:
    """Result of a single validation run."""

    passed: bool
    stage_reached: str
    stages: dict[str, bool] = field(default_factory=dict)
    error: str = ""
    error_stage: str = ""
    frame_shape: tuple[int, ...] | None = None
    rollout_rewards: list[float] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary for the agent."""
        lines = ["Validation Result:"]
        for stage, ok in self.stages.items():
            icon = "PASS" if ok else "FAIL"
            lines.append(f"  [{icon}] {stage}")
        if self.error:
            lines.append(f"\nError at stage '{self.error_stage}':")
            lines.append(self.error)
        if self.rollout_rewards:
            lines.append(
                f"\nRollout: {len(self.rollout_rewards)} steps, "
                f"total reward={sum(self.rollout_rewards):.4f}"
            )
        if self.frame_shape:
            lines.append(f"Frame shape: {self.frame_shape}")
        return "\n".join(lines)


def _purge_env_modules(env_dir: Path) -> None:
    """Remove all cached modules for this env so we get fresh imports."""
    prefix = f"envs.{env_dir.name}"
    for key in list(sys.modules.keys()):
        if key == prefix or key.startswith(prefix + "."):
            del sys.modules[key]
    if "envs" in sys.modules:
        del sys.modules["envs"]

    pycache = env_dir / "__pycache__"
    if pycache.exists():
        import shutil

        shutil.rmtree(pycache, ignore_errors=True)


def _setup_import_paths(env_dir: Path) -> str:
    """Set up sys.path and package structure for relative imports.

    Returns the fully-qualified module name for env.py.
    """
    root_str = str(env_dir.parent.parent)
    env_dir_str = str(env_dir)
    for p_str in (root_str, env_dir_str):
        if p_str not in sys.path:
            sys.path.insert(0, p_str)

    envs_pkg_name = "envs"
    env_pkg_name = f"envs.{env_dir.name}"

    if envs_pkg_name not in sys.modules:
        envs_pkg = type(sys)(envs_pkg_name)
        envs_pkg.__path__ = [str(env_dir.parent)]
        envs_pkg.__package__ = envs_pkg_name
        sys.modules[envs_pkg_name] = envs_pkg

    if env_pkg_name not in sys.modules:
        env_pkg = type(sys)(env_pkg_name)
        env_pkg.__path__ = [env_dir_str]
        env_pkg.__package__ = env_pkg_name
        sys.modules[env_pkg_name] = env_pkg

    for sibling in env_dir.glob("*.py"):
        if sibling.name in ("env.py", "__init__.py"):
            continue
        sib_mod_name = f"envs.{env_dir.name}.{sibling.stem}"
        if sib_mod_name not in sys.modules:
            sib_spec = importlib.util.spec_from_file_location(
                sib_mod_name, str(sibling)
            )
            if sib_spec and sib_spec.loader:
                sib_mod = importlib.util.module_from_spec(sib_spec)
                sib_mod.__package__ = env_pkg_name
                sys.modules[sib_mod_name] = sib_mod
                try:
                    sib_spec.loader.exec_module(sib_mod)
                except Exception:
                    pass

    return f"envs.{env_dir.name}.env"


def _force_headless() -> None:
    """Force headless mode for Pygame so validation doesn't open windows."""
    import os
    import platform

    os.environ["SDL_VIDEODRIVER"] = "dummy"
    if platform.system() == "Linux":
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["MUJOCO_GL"] = "egl"
    else:
        os.environ.pop("PYOPENGL_PLATFORM", None)
        os.environ.pop("MUJOCO_GL", None)


def _is_genesis_env(env_py: Path) -> bool:
    """Quick check if env.py uses Genesis (affects validation strategy)."""
    source = env_py.read_text(encoding="utf-8")
    return "import genesis" in source or "from genesis" in source


def _validate_via_subprocess(env_dir: Path, rollout_steps: int) -> ValidationResult:
    """Run validation in an isolated subprocess (required for Genesis on macOS)."""
    try:
        proc = subprocess.run(
            [sys.executable, str(_VALIDATE_SCRIPT), str(env_dir), str(rollout_steps)],
            capture_output=True,
            text=True,
            timeout=_GENESIS_SUBPROCESS_TIMEOUT,
            cwd=str(env_dir.parent.parent),
        )
    except subprocess.TimeoutExpired:
        return ValidationResult(
            passed=False,
            stage_reached="timeout",
            stages={},
            error=f"Validation timed out after {_GENESIS_SUBPROCESS_TIMEOUT}s",
            error_stage="timeout",
        )
    except Exception:
        return ValidationResult(
            passed=False,
            stage_reached="subprocess",
            stages={},
            error=traceback.format_exc(),
            error_stage="subprocess",
        )

    stdout = proc.stdout.strip()
    if proc.returncode != 0 or not stdout:
        stderr = proc.stderr.strip()
        return ValidationResult(
            passed=False,
            stage_reached="subprocess",
            stages={},
            error=f"Subprocess exited {proc.returncode}:\n{stderr}\n{stdout}",
            error_stage="subprocess",
        )

    try:
        data = json.loads(stdout.split("\n")[-1])
    except (json.JSONDecodeError, IndexError):
        return ValidationResult(
            passed=False,
            stage_reached="subprocess",
            stages={},
            error=f"Could not parse subprocess output:\n{stdout}",
            error_stage="subprocess",
        )

    return ValidationResult(
        passed=data.get("passed", False),
        stage_reached=data.get("stage_reached", "unknown"),
        stages=data.get("stages", {}),
        error=data.get("error", ""),
        error_stage=data.get("error_stage", ""),
        frame_shape=tuple(data["frame_shape"]) if data.get("frame_shape") else None,
        rollout_rewards=data.get("rollout_rewards", []),
    )


def _validate_env_dir(env_dir: Path, rollout_steps: int) -> ValidationResult:
    """Run the full 8-stage validation pipeline on an environment directory.

    Genesis envs are validated in a subprocess to avoid macOS OpenGL crashes.
    """
    env_py = env_dir / "env.py"
    is_genesis = _is_genesis_env(env_py) if env_py.exists() else False

    if is_genesis:
        return _validate_via_subprocess(env_dir, rollout_steps)

    _purge_env_modules(env_dir)
    _force_headless()

    stages: dict[str, bool] = {}

    # ── Stage 1: file_check ──────────────────────────────────────────────
    if not env_py.exists():
        return ValidationResult(
            passed=False,
            stage_reached="file_check",
            stages={"file_check": False},
            error=f"env.py not found in {env_dir}",
            error_stage="file_check",
        )
    stages["file_check"] = True

    # ── Stage 2: syntax ──────────────────────────────────────────────────
    source = env_py.read_text(encoding="utf-8")
    try:
        compile(source, str(env_py), "exec")
        stages["syntax"] = True
    except SyntaxError:
        stages["syntax"] = False
        return ValidationResult(
            passed=False,
            stage_reached="syntax",
            stages=stages,
            error=traceback.format_exc(),
            error_stage="syntax",
        )

    # ── Stage 3: import ──────────────────────────────────────────────────
    module_name = _setup_import_paths(env_dir)
    env_pkg_name = f"envs.{env_dir.name}"

    spec = importlib.util.spec_from_file_location(module_name, str(env_py))
    if spec is None or spec.loader is None:
        stages["import"] = False
        return ValidationResult(
            passed=False,
            stage_reached="import",
            stages=stages,
            error=f"Could not create module spec for {env_py}",
            error_stage="import",
        )

    module = importlib.util.module_from_spec(spec)
    module.__package__ = env_pkg_name
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        stages["import"] = True
    except Exception:
        stages["import"] = False
        return ValidationResult(
            passed=False,
            stage_reached="import",
            stages=stages,
            error=traceback.format_exc(),
            error_stage="import",
        )

    env_class = _find_env_class(module)
    if env_class is None:
        stages["import"] = False
        return ValidationResult(
            passed=False,
            stage_reached="import",
            stages=stages,
            error="No class inheriting gymnasium.Env found in env.py",
            error_stage="import",
        )

    # ── Stage 4: instantiate ─────────────────────────────────────────────
    env_instance = None
    try:
        init_kwargs: dict[str, Any] = {}
        if is_genesis:
            init_kwargs["show_viewer"] = False
        env_instance = env_class(**init_kwargs)
        stages["instantiate"] = True
    except TypeError:
        try:
            env_instance = env_class()
            stages["instantiate"] = True
        except Exception:
            stages["instantiate"] = False
            return ValidationResult(
                passed=False,
                stage_reached="instantiate",
                stages=stages,
                error=traceback.format_exc(),
                error_stage="instantiate",
            )
    except Exception:
        stages["instantiate"] = False
        return ValidationResult(
            passed=False,
            stage_reached="instantiate",
            stages=stages,
            error=traceback.format_exc(),
            error_stage="instantiate",
        )

    # ── Stage 5: reset — check observation shape/dtype ───────────────────
    try:
        obs, info = env_instance.reset()
        obs_arr = np.asarray(obs)
        if not env_instance.observation_space.contains(obs):
            raise ValueError(
                f"Reset obs shape {obs_arr.shape} dtype {obs_arr.dtype} "
                f"not in observation_space {env_instance.observation_space}"
            )
        stages["reset"] = True
    except Exception:
        stages["reset"] = False
        return _fail(stages, "reset", env_instance)

    # ── Stage 6: step — N random actions, check rewards/dones/obs ────────
    rewards: list[float] = []
    try:
        for _ in range(rollout_steps):
            action = env_instance.action_space.sample()
            obs, reward, terminated, truncated, info = env_instance.step(action)
            rewards.append(float(reward))
            if not isinstance(terminated, (bool, np.bool_)):
                raise TypeError(f"terminated must be bool, got {type(terminated)}")
            if not isinstance(truncated, (bool, np.bool_)):
                raise TypeError(f"truncated must be bool, got {type(truncated)}")
            if terminated or truncated:
                obs, info = env_instance.reset()
        stages["step"] = True
    except Exception:
        stages["step"] = False
        return _fail(stages, "step", env_instance, rewards=rewards)

    if any(np.isnan(r) or np.isinf(r) for r in rewards):
        stages["step"] = False
        return ValidationResult(
            passed=False,
            stage_reached="step",
            stages=stages,
            error=f"NaN or Inf reward detected in {len(rewards)} steps",
            error_stage="step",
            rollout_rewards=rewards,
        )

    # ── Stage 7: render — one RGB frame ──────────────────────────────────
    frame_shape = None
    try:
        env_instance.close()
        render_kwargs: dict[str, Any] = {"render_mode": "rgb_array"}
        if is_genesis:
            render_kwargs["show_viewer"] = False
        try:
            render_env = env_class(**render_kwargs)
        except TypeError:
            render_env = env_class(render_mode="rgb_array")
        render_env.reset()
        frame = render_env.render()
        if frame is not None:
            frame = np.asarray(frame)
            frame_shape = frame.shape
        render_env.close()
        stages["render"] = True
    except Exception:
        stages["render"] = False
        return _fail(stages, "render", None, rewards=rewards)

    # ── Stage 8: check_env (SB3) ─────────────────────────────────────────
    check_env_instance = None
    try:
        from stable_baselines3.common.env_checker import check_env

        check_kwargs: dict[str, Any] = {}
        if is_genesis:
            check_kwargs["show_viewer"] = False
        try:
            check_env_instance = env_class(**check_kwargs)
        except TypeError:
            check_env_instance = env_class()
        check_env(check_env_instance, warn=True, skip_render_check=True)
        check_env_instance.close()
        stages["check_env"] = True
    except Exception:
        stages["check_env"] = False
        if check_env_instance:
            try:
                check_env_instance.close()
            except Exception:
                pass
        return ValidationResult(
            passed=False,
            stage_reached="check_env",
            stages=stages,
            error=traceback.format_exc(),
            error_stage="check_env",
            rollout_rewards=rewards,
        )

    return ValidationResult(
        passed=True,
        stage_reached="check_env",
        stages=stages,
        frame_shape=frame_shape,
        rollout_rewards=rewards,
    )


def _find_env_class(module: Any) -> type | None:
    """Find the first class in module that inherits from gymnasium.Env."""
    import gymnasium

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, gymnasium.Env)
            and attr is not gymnasium.Env
        ):
            return attr
    return None


def _fail(
    stages: dict[str, bool],
    stage: str,
    env_instance: Any,
    rewards: list[float] | None = None,
) -> ValidationResult:
    """Build a failure result, closing the env safely."""
    try:
        if env_instance is not None:
            env_instance.close()
    except Exception:
        pass
    return ValidationResult(
        passed=False,
        stage_reached=stage,
        stages=stages,
        error=traceback.format_exc(),
        error_stage=stage,
        rollout_rewards=rewards or [],
    )


# ── LangChain tool wrapper ──────────────────────────────────────────────────


@tool
def eval_env(env_id: str, rollout_steps: int = VALIDATION_ROLLOUT_STEPS) -> str:
    """Validate a generated environment through the full pipeline.

    Runs: file_check → syntax → import → instantiate → reset → step → render → check_env.

    Args:
        env_id: The environment directory name under envs/.
        rollout_steps: Number of random rollout steps (default from config).
    """
    env_dir = ENVS_DIR / env_id
    if not env_dir.is_dir():
        return f"ERROR: Environment directory not found: envs/{env_id}"

    result = _validate_env_dir(env_dir, rollout_steps)
    return result.summary()
