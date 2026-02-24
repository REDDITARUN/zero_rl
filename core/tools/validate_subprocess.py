"""Standalone validation script run as a subprocess.

Handles both Gymnasium and native Genesis environments.
Genesis envs are plain classes with tensor-based step/reset.
Outputs JSON with ValidationResult fields to stdout.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

import platform

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
if platform.system() == "Linux":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_GL", "egl")
else:
    os.environ.pop("PYOPENGL_PLATFORM", None)
    os.environ.pop("MUJOCO_GL", None)

import numpy as np


def _setup_import_paths(env_dir: Path) -> str:
    """Set up sys.path and package structure for relative imports."""
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


def _is_genesis_env(env_py: Path) -> bool:
    source = env_py.read_text(encoding="utf-8")
    return "import genesis" in source or "from genesis" in source


def _find_gym_class(module):
    """Find the first gymnasium.Env subclass."""
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


def _find_genesis_class(module):
    """Find the main env class for a native Genesis env.

    Heuristic: a class defined in the module (not imported) that has
    both reset() and step() methods.
    """
    candidates = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if not isinstance(attr, type):
            continue
        if getattr(attr, "__module__", None) != module.__name__:
            continue
        if hasattr(attr, "reset") and hasattr(attr, "step"):
            candidates.append(attr)
    if len(candidates) == 1:
        return candidates[0]
    for cls in candidates:
        if "env" in cls.__name__.lower():
            return cls
    return candidates[0] if candidates else None


def _load_genesis_configs(env_dir: Path) -> dict:
    """Load config dicts from config.py (get_env_cfg, get_obs_cfg, etc)."""
    config_py = env_dir / "config.py"
    if not config_py.exists():
        return {}

    pkg_name = f"envs.{env_dir.name}"
    mod_name = f"{pkg_name}.config"
    if mod_name in sys.modules:
        cfg_mod = sys.modules[mod_name]
    else:
        return {}

    configs = {}
    for fn_name in ("get_env_cfg", "get_obs_cfg", "get_reward_cfg", "get_command_cfg"):
        fn = getattr(cfg_mod, fn_name, None)
        if callable(fn):
            try:
                configs[fn_name.replace("get_", "").replace("_cfg", "_cfg")] = fn()
            except Exception:
                pass
    return configs


def _instantiate_genesis(env_class, configs: dict):
    """Try to instantiate a native Genesis env with configs."""
    sig = inspect.signature(env_class.__init__)
    params = list(sig.parameters.keys())

    kwargs: dict = {"show_viewer": False}
    if "num_envs" in params:
        kwargs["num_envs"] = 1

    param_to_config = {
        "env_cfg": "env_cfg",
        "obs_cfg": "obs_cfg",
        "reward_cfg": "reward_cfg",
        "command_cfg": "command_cfg",
    }
    for param, cfg_key in param_to_config.items():
        if param in params and cfg_key in configs:
            kwargs[param] = configs[cfg_key]

    return env_class(**kwargs)


def validate(env_dir_str: str, rollout_steps: int) -> dict:
    """Run the validation pipeline, return dict serializable to JSON."""
    env_dir = Path(env_dir_str)
    env_py = env_dir / "env.py"
    is_genesis = _is_genesis_env(env_py) if env_py.exists() else False
    stages: dict[str, bool] = {}

    def fail(stage: str, error: str, rewards=None) -> dict:
        return {
            "passed": False,
            "stage_reached": stage,
            "stages": stages,
            "error": error,
            "error_stage": stage,
            "frame_shape": None,
            "rollout_rewards": rewards or [],
        }

    # Stage 1: file_check
    if not env_py.exists():
        stages["file_check"] = False
        return fail("file_check", f"env.py not found in {env_dir}")
    stages["file_check"] = True

    # Stage 2: syntax
    source = env_py.read_text(encoding="utf-8")
    try:
        compile(source, str(env_py), "exec")
        stages["syntax"] = True
    except SyntaxError:
        stages["syntax"] = False
        return fail("syntax", traceback.format_exc())

    # Stage 3: import
    module_name = _setup_import_paths(env_dir)
    env_pkg_name = f"envs.{env_dir.name}"

    spec = importlib.util.spec_from_file_location(module_name, str(env_py))
    if spec is None or spec.loader is None:
        stages["import"] = False
        return fail("import", f"Could not create module spec for {env_py}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = env_pkg_name
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        stages["import"] = True
    except Exception:
        stages["import"] = False
        return fail("import", traceback.format_exc())

    gym_class = _find_gym_class(module)
    if gym_class is not None:
        return _validate_gymnasium(module, env_dir, stages, rollout_steps, fail)
    elif is_genesis:
        return _validate_genesis(module, env_dir, stages, rollout_steps, fail)
    else:
        stages["import"] = False
        return fail("import", "No gymnasium.Env subclass or Genesis env class found in env.py")


def _validate_gymnasium(module, env_dir, stages, rollout_steps, fail) -> dict:
    """Gymnasium env validation: standard 8-stage pipeline."""
    env_class = _find_gym_class(module)
    if env_class is None:
        stages["import"] = False
        return fail("import", "No class inheriting gymnasium.Env found in env.py")

    # Stage 4: instantiate
    try:
        env_instance = env_class()
        stages["instantiate"] = True
    except Exception:
        stages["instantiate"] = False
        return fail("instantiate", traceback.format_exc())

    # Stage 5: reset
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
        try:
            env_instance.close()
        except Exception:
            pass
        return fail("reset", traceback.format_exc())

    # Stage 6: step
    rewards: list[float] = []
    try:
        for _ in range(rollout_steps):
            action = env_instance.action_space.sample()
            obs, reward, terminated, truncated, info = env_instance.step(action)
            rewards.append(float(reward))
            if terminated or truncated:
                obs, info = env_instance.reset()
        stages["step"] = True
    except Exception:
        stages["step"] = False
        try:
            env_instance.close()
        except Exception:
            pass
        return fail("step", traceback.format_exc(), rewards)

    if any(np.isnan(r) or np.isinf(r) for r in rewards):
        stages["step"] = False
        try:
            env_instance.close()
        except Exception:
            pass
        return fail("step", f"NaN or Inf reward in {len(rewards)} steps", rewards)

    # Stage 7: render
    frame_shape = None
    try:
        env_instance.close()
        try:
            render_env = type(env_instance)(render_mode="rgb_array")
        except TypeError:
            render_env = type(env_instance)()
        render_env.reset()
        frame = render_env.render()
        if frame is not None:
            frame = np.asarray(frame)
            frame_shape = list(frame.shape)
        render_env.close()
        stages["render"] = True
    except Exception:
        stages["render"] = False
        return fail("render", traceback.format_exc())

    # Stage 8: check_env (SB3)
    try:
        from stable_baselines3.common.env_checker import check_env
        check_inst = type(env_instance)()
        check_env(check_inst, warn=True, skip_render_check=True)
        check_inst.close()
        stages["check_env"] = True
    except Exception:
        stages["check_env"] = False
        return {
            "passed": False,
            "stage_reached": "check_env",
            "stages": stages,
            "error": traceback.format_exc(),
            "error_stage": "check_env",
            "frame_shape": frame_shape,
            "rollout_rewards": rewards,
        }

    return {
        "passed": True,
        "stage_reached": "check_env",
        "stages": stages,
        "error": "",
        "error_stage": "",
        "frame_shape": frame_shape,
        "rollout_rewards": rewards,
    }


def _validate_genesis(module, env_dir, stages, rollout_steps, fail) -> dict:
    """Native Genesis env validation: no gymnasium.Env, tensor interface.

    Stages: file_check, syntax, import, instantiate, reset, step.
    No render or check_env stages (Genesis uses its own viewer).
    """
    import torch

    env_class = _find_genesis_class(module)
    if env_class is None:
        stages["import"] = False
        return fail("import", "No env class with reset()/step() found in env.py")

    configs = _load_genesis_configs(env_dir)

    # Stage 4: instantiate
    env = None
    try:
        env = _instantiate_genesis(env_class, configs)
        stages["instantiate"] = True
    except Exception:
        stages["instantiate"] = False
        return fail("instantiate", traceback.format_exc())

    # Stage 5: reset
    try:
        result = env.reset()
        if isinstance(result, tuple):
            obs_buf = result[0]
        else:
            obs_buf = result

        if not isinstance(obs_buf, torch.Tensor):
            raise TypeError(f"reset() obs must be a torch.Tensor, got {type(obs_buf)}")
        if obs_buf.dim() < 1:
            raise ValueError(f"reset() obs must be at least 1D, got shape {obs_buf.shape}")
        stages["reset"] = True
    except Exception:
        stages["reset"] = False
        return fail("reset", traceback.format_exc())

    # Stage 6: step
    rewards_sum = 0.0
    try:
        num_actions = getattr(env, "num_actions", None)
        num_envs = getattr(env, "num_envs", 1)
        device = getattr(env, "device", "cpu")

        if num_actions is None:
            raise ValueError("Env must define num_actions attribute")

        for i in range(rollout_steps):
            actions = torch.randn(num_envs, num_actions, device=device)
            result = env.step(actions)

            if not isinstance(result, tuple) or len(result) < 3:
                raise TypeError(
                    f"step() must return (obs, rew, reset, ...), got {type(result)} len={len(result) if isinstance(result, tuple) else 'N/A'}"
                )

            obs_buf, rew_buf, reset_buf = result[0], result[1], result[2]

            if not isinstance(rew_buf, torch.Tensor):
                raise TypeError(f"step() rew_buf must be torch.Tensor, got {type(rew_buf)}")

            r = float(rew_buf.mean())
            if np.isnan(r) or np.isinf(r):
                raise ValueError(f"NaN/Inf reward at step {i}: {r}")
            rewards_sum += r

        stages["step"] = True
    except Exception:
        stages["step"] = False
        return fail("step", traceback.format_exc())

    return {
        "passed": True,
        "stage_reached": "step",
        "stages": stages,
        "error": "",
        "error_stage": "",
        "frame_shape": None,
        "rollout_rewards": [rewards_sum / max(rollout_steps, 1)],
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: validate_subprocess.py <env_dir> [rollout_steps]"}))
        sys.exit(1)

    env_dir_arg = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    try:
        result = validate(env_dir_arg, steps)
    except Exception:
        result = {
            "passed": False,
            "stage_reached": "unknown",
            "stages": {},
            "error": traceback.format_exc(),
            "error_stage": "unknown",
            "frame_shape": None,
            "rollout_rewards": [],
        }

    print(json.dumps(result))
