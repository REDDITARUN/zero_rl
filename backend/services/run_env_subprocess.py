"""Standalone script to run an env and emit JSON-line frames to stdout.

Handles both Gymnasium and native Genesis environments.
For Genesis: dynamically adds a camera for headless frame capture,
and uses the native tensor-based step/reset interface.

Camera commands are read from stdin as JSON lines:
  {"camera": {"orbit": [dx, dy]}}  — turntable orbit (normalised viewport fractions)
  {"camera": {"zoom": dz}}         — dolly zoom (+1 = in, -1 = out)
  {"camera": {"pan": [dx, dy]}}    — screen-space pan (normalised fractions)
  {"camera": {"reset": true}}      — reset to default view
"""

from __future__ import annotations

import base64
import importlib
import importlib.util
import inspect
import io
import json
import math
import os
import select
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

_real_stdout = sys.stdout
_real_stdin = sys.stdin


def emit(data: dict) -> None:
    """Write one JSON line to the real stdout (fd 1)."""
    _real_stdout.write(json.dumps(data) + "\n")
    _real_stdout.flush()


def _read_stdin_commands() -> list[dict]:
    """Non-blocking read of JSON commands from stdin."""
    commands = []
    try:
        while select.select([_real_stdin], [], [], 0)[0]:
            line = _real_stdin.readline()
            if not line:
                break
            line = line.strip()
            if line:
                try:
                    commands.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass
    return commands


class CameraController:
    """Turntable orbit camera — horizontal drag rotates around world Z,
    vertical drag tilts elevation. Feels like Blender / Three.js OrbitControls.

    The frontend sends normalised deltas (-1..+1 range, fraction of viewport).
    """

    MIN_RADIUS = 0.3
    MAX_RADIUS = 30.0
    MIN_ELEVATION = 0.05          # just above the ground plane
    MAX_ELEVATION = math.pi * 0.48  # almost straight down but not quite

    def __init__(self, camera, init_pos: tuple, init_lookat: tuple) -> None:
        self.camera = camera
        self.default_pos = np.array(init_pos, dtype=np.float64)
        self.default_lookat = np.array(init_lookat, dtype=np.float64)

        # Decompose initial pose into spherical coordinates
        offset = self.default_pos - self.default_lookat
        self._radius = float(np.linalg.norm(offset))
        self._azimuth = float(math.atan2(offset[1], offset[0]))
        self._elevation = float(math.acos(np.clip(offset[2] / max(self._radius, 1e-6), -1, 1)))
        self._target = self.default_lookat.copy()

        self._dirty = False

    def _pos_from_spherical(self) -> np.ndarray:
        """Reconstruct camera position from spherical coords + target."""
        sp = math.sin(self._elevation)
        return self._target + np.array([
            self._radius * sp * math.cos(self._azimuth),
            self._radius * sp * math.sin(self._azimuth),
            self._radius * math.cos(self._elevation),
        ])

    def orbit(self, dx: float, dy: float) -> None:
        """Turntable orbit. dx/dy are normalised viewport fractions.

        dx > 0 = drag right = scene rotates right (camera goes left).
        dy > 0 = drag down  = camera tilts upward (looking more from above).
        """
        self._azimuth -= dx * math.pi      # full drag across viewport = 180°
        self._elevation = np.clip(
            self._elevation + dy * math.pi * 0.5,
            self.MIN_ELEVATION,
            self.MAX_ELEVATION,
        )
        self._dirty = True

    def zoom(self, dz: float) -> None:
        """Dolly zoom. dz > 0 = zoom in, dz < 0 = zoom out."""
        factor = 1.0 - dz * 0.12
        self._radius = float(np.clip(self._radius * factor, self.MIN_RADIUS, self.MAX_RADIUS))
        self._dirty = True

    def pan(self, dx: float, dy: float) -> None:
        """Screen-space pan. dx/dy are normalised viewport fractions."""
        pos = self._pos_from_spherical()
        forward = self._target - pos
        forward /= max(np.linalg.norm(forward), 1e-6)
        right = np.cross(forward, np.array([0, 0, 1]))
        rn = np.linalg.norm(right)
        if rn > 1e-6:
            right /= rn
        up = np.cross(right, forward)

        scale = self._radius * 0.8
        delta = -right * dx * scale + up * dy * scale
        self._target += delta
        self._dirty = True

    def reset(self) -> None:
        """Reset to initial view."""
        offset = self.default_pos - self.default_lookat
        self._radius = float(np.linalg.norm(offset))
        self._azimuth = float(math.atan2(offset[1], offset[0]))
        self._elevation = float(math.acos(np.clip(offset[2] / max(self._radius, 1e-6), -1, 1)))
        self._target = self.default_lookat.copy()
        self._dirty = True

    def apply(self) -> None:
        """Push camera pose to Genesis renderer."""
        if not self._dirty:
            return
        pos = self._pos_from_spherical()
        try:
            self.camera.set_pose(
                pos=tuple(pos.tolist()),
                lookat=tuple(self._target.tolist()),
            )
        except Exception:
            pass
        self._dirty = False

    def process_commands(self, commands: list[dict]) -> None:
        """Process a batch of camera commands from the frontend."""
        for cmd in commands:
            cam = cmd.get("camera")
            if not cam:
                continue
            if "orbit" in cam:
                ox, oy = cam["orbit"]
                self.orbit(float(ox), float(oy))
            if "zoom" in cam:
                self.zoom(float(cam["zoom"]))
            if "pan" in cam:
                px, py = cam["pan"]
                self.pan(float(px), float(py))
            if cam.get("reset"):
                self.reset()
        self.apply()


def _setup_import_paths(env_dir: Path) -> str:
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
            sib_spec = importlib.util.spec_from_file_location(sib_mod_name, str(sibling))
            if sib_spec and sib_spec.loader:
                sib_mod = importlib.util.module_from_spec(sib_spec)
                sib_mod.__package__ = env_pkg_name
                sys.modules[sib_mod_name] = sib_mod
                try:
                    sib_spec.loader.exec_module(sib_mod)
                except Exception:
                    pass

    return f"envs.{env_dir.name}.env"


def _is_genesis_env(source: str) -> bool:
    return "import genesis" in source or "from genesis" in source


def _find_gym_class(module):
    import gymnasium
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, gymnasium.Env) and attr is not gymnasium.Env:
            return attr
    return None


def _find_genesis_class(module):
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
    pkg_name = f"envs.{env_dir.name}"
    mod_name = f"{pkg_name}.config"
    cfg_mod = sys.modules.get(mod_name)
    if cfg_mod is None:
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


def _render_frame_b64(frame) -> str:
    """Convert a frame (numpy array or tuple from camera.render) to base64 PNG."""
    if isinstance(frame, tuple):
        frame = frame[0]
    if frame is None:
        return ""
    if hasattr(frame, "cpu"):
        frame = frame.cpu().numpy()
    arr = np.asarray(frame)
    if arr.ndim < 2:
        return ""
    if arr.ndim == 4:
        arr = arr[0]
    try:
        from PIL import Image
        img = Image.fromarray(arr.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


_injected_camera = None
_injected_cam_pos = (3.0, -2.0, 2.5)
_injected_cam_lookat = (0.0, 0.0, 0.5)

_DEFAULT_LIGHTS = [
    {"type": "directional", "dir": (-1, -0.8, -1.2), "color": (1.0, 0.98, 0.95), "intensity": 5.5},
    {"type": "directional", "dir": (0.6, 0.3, -0.8), "color": (0.45, 0.55, 0.75), "intensity": 2.5},
    {"type": "directional", "dir": (0.2, -0.6, -0.4), "color": (0.40, 0.38, 0.45), "intensity": 1.5},
]


def _patch_scene_build():
    """Monkey-patch gs.Scene.build to inject a camera and visual defaults."""
    import genesis as gs

    original_build = gs.Scene.build

    def patched_build(self, *args, **kwargs):
        global _injected_camera, _injected_cam_pos, _injected_cam_lookat

        cam_pos = (3.0, -2.0, 2.5)
        cam_lookat = (0.0, 0.0, 0.5)
        cam_fov = 40

        vo = getattr(self, "_viewer_options", None) or getattr(self, "viewer_options", None)
        if vo is not None:
            cam_pos = getattr(vo, "camera_pos", cam_pos)
            cam_lookat = getattr(vo, "camera_lookat", cam_lookat)
            cam_fov = getattr(vo, "camera_fov", cam_fov)

        vis = getattr(self, "_vis_options", None) or getattr(self, "vis_options", None)
        if vis is not None:
            default_single_light = [{"type": "directional", "dir": (-1, -1, -1), "color": (1.0, 1.0, 1.0), "intensity": 5.0}]
            if getattr(vis, "lights", None) == default_single_light or getattr(vis, "lights", None) is None:
                try:
                    vis.lights = _DEFAULT_LIGHTS
                except Exception:
                    pass
            if getattr(vis, "ambient_light", None) == (0.1, 0.1, 0.1):
                try:
                    vis.ambient_light = (0.25, 0.25, 0.28)
                except Exception:
                    pass
            if getattr(vis, "background_color", None) == (0.04, 0.08, 0.12):
                try:
                    vis.background_color = (0.14, 0.16, 0.20)
                except Exception:
                    pass
            try:
                vis.shadow = True
            except Exception:
                pass

        _injected_cam_pos = cam_pos
        _injected_cam_lookat = cam_lookat

        _injected_camera = self.add_camera(
            res=(960, 720),
            pos=cam_pos,
            lookat=cam_lookat,
            fov=cam_fov,
            GUI=False,
            spp=512,
        )

        return original_build(self, *args, **kwargs)

    gs.Scene.build = patched_build


def run_gymnasium(env_cls, max_steps: int) -> None:
    """Run a Gymnasium env and emit frames."""
    env = env_cls(render_mode="rgb_array")
    try:
        obs, info = env.reset()
        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            frame_b64 = _render_frame_b64(env.render())

            safe_info = {}
            for k, v in (info or {}).items():
                if isinstance(v, (int, float, str, np.floating)):
                    safe_info[k] = float(v) if isinstance(v, (int, float, np.floating)) else str(v)

            emit({
                "step": step,
                "frame": frame_b64,
                "reward": float(reward),
                "done": bool(terminated or truncated),
                "actions": action.tolist() if hasattr(action, "tolist") else [int(action)],
                "info": safe_info,
            })

            if terminated or truncated:
                obs, info = env.reset()
    finally:
        env.close()


def run_genesis(env_cls, env_dir: Path, max_steps: int) -> None:
    """Run a native Genesis env and emit frames with interactive camera."""
    global _injected_camera, _injected_cam_pos, _injected_cam_lookat
    import torch

    _patch_scene_build()

    configs = _load_genesis_configs(env_dir)
    sig = inspect.signature(env_cls.__init__)
    params = list(sig.parameters.keys())

    kwargs: dict = {"show_viewer": False}
    if "num_envs" in params:
        kwargs["num_envs"] = 1
    for param in ("env_cfg", "obs_cfg", "reward_cfg", "command_cfg"):
        if param in params and param in configs:
            kwargs[param] = configs[param]

    env = env_cls(**kwargs)

    camera = _injected_camera
    if camera is None:
        emit({"error": "Failed to inject camera into Genesis scene"})
        return

    cam_ctrl = CameraController(camera, _injected_cam_pos, _injected_cam_lookat)

    num_actions = getattr(env, "num_actions", 4)
    num_envs = getattr(env, "num_envs", 1)
    device = getattr(env, "device", "cpu")

    try:
        env.reset()
        for step in range(max_steps):
            stdin_cmds = _read_stdin_commands()
            cam_ctrl.process_commands(stdin_cmds)

            actions = torch.randn(num_envs, num_actions, device=device)
            result = env.step(actions)
            rew_buf = result[1] if isinstance(result, tuple) and len(result) > 1 else torch.zeros(1)
            reset_buf = result[2] if isinstance(result, tuple) and len(result) > 2 else torch.zeros(1, dtype=torch.bool)

            frame_b64 = _render_frame_b64(camera.render(rgb=True))

            emit({
                "step": step,
                "frame": frame_b64,
                "reward": float(rew_buf.mean()),
                "done": bool(reset_buf.any()),
                "actions": actions[0].tolist() if actions.dim() > 1 else actions.tolist(),
                "info": {},
            })

            if reset_buf.any():
                env.reset()
    except Exception:
        emit({"error": traceback.format_exc()})


def run(env_dir_str: str, max_steps: int) -> None:
    env_dir = Path(env_dir_str)
    env_py = env_dir / "env.py"
    source = env_py.read_text(encoding="utf-8")
    is_genesis = _is_genesis_env(source)

    if is_genesis:
        sys.stdout = sys.stderr

    module_name = _setup_import_paths(env_dir)
    spec = importlib.util.spec_from_file_location(module_name, str(env_py))
    if spec is None or spec.loader is None:
        emit({"error": "Could not create module spec"})
        return

    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = f"envs.{env_dir.name}"
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    gym_cls = _find_gym_class(mod)
    if gym_cls is not None:
        run_gymnasium(gym_cls, max_steps)
    elif is_genesis:
        genesis_cls = _find_genesis_class(mod)
        if genesis_cls is None:
            emit({"error": "No Genesis env class found (needs reset + step methods)"})
            return
        run_genesis(genesis_cls, env_dir, max_steps)
    else:
        emit({"error": "No gymnasium.Env subclass or Genesis env class found"})


if __name__ == "__main__":
    if len(sys.argv) < 2:
        emit({"error": "Usage: run_env_subprocess.py <env_dir> [max_steps]"})
        sys.exit(1)

    env_dir_arg = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    try:
        run(env_dir_arg, steps)
    except Exception:
        emit({"error": traceback.format_exc()})
