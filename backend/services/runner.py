"""Environment runner — execute an env and stream render frames + rewards.

Genesis envs are run in a subprocess to avoid macOS OpenGL thread crashes.
Pygame/Gymnasium envs run in-process for lower latency.

Camera commands (orbit, zoom, move) are forwarded to subprocess via stdin.
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import AsyncIterator

import numpy as np

from core.config import ENVS_DIR

_RUN_SCRIPT = Path(__file__).parent / "run_env_subprocess.py"


def _is_genesis_env(env_dir: Path) -> bool:
    source = (env_dir / "env.py").read_text(encoding="utf-8")
    return "import genesis" in source or "from genesis" in source


def _purge_modules(env_dir: Path) -> None:
    """Remove cached env modules for a clean import."""
    prefix = f"envs.{env_dir.name}"
    for key in list(sys.modules.keys()):
        if key == prefix or key.startswith(prefix + "."):
            del sys.modules[key]
    if "envs" in sys.modules:
        del sys.modules["envs"]


def _load_env_class(env_dir: Path) -> type | None:
    """Import env.py and find the Gymnasium Env subclass."""
    import gymnasium

    _purge_modules(env_dir)

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
        sib_name = f"envs.{env_dir.name}.{sibling.stem}"
        if sib_name not in sys.modules:
            sib_spec = importlib.util.spec_from_file_location(sib_name, str(sibling))
            if sib_spec and sib_spec.loader:
                sib_mod = importlib.util.module_from_spec(sib_spec)
                sib_mod.__package__ = env_pkg_name
                sys.modules[sib_name] = sib_mod
                try:
                    sib_spec.loader.exec_module(sib_mod)
                except Exception:
                    pass

    module_name = f"envs.{env_dir.name}.env"
    env_py = env_dir / "env.py"
    spec = importlib.util.spec_from_file_location(module_name, str(env_py))
    if spec is None or spec.loader is None:
        return None

    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = env_pkg_name
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, gymnasium.Env)
            and attr is not gymnasium.Env
        ):
            return attr
    return None


async def _stream_via_subprocess(
    env_dir: Path, max_steps: int, camera_queue: asyncio.Queue | None = None
) -> AsyncIterator[dict]:
    """Run the env in a subprocess and stream JSON-line frames.

    Camera commands from the queue are forwarded to the subprocess via stdin.
    """
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(_RUN_SCRIPT),
        str(env_dir),
        str(max_steps),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE,
        cwd=str(env_dir.parent.parent),
        limit=1024 * 1024,
    )
    assert proc.stdout is not None

    async def _forward_camera():
        """Forward camera commands from queue to subprocess stdin."""
        if camera_queue is None or proc.stdin is None:
            return
        try:
            while True:
                cmd = await camera_queue.get()
                line = json.dumps(cmd) + "\n"
                proc.stdin.write(line.encode())
                await proc.stdin.drain()
        except (asyncio.CancelledError, BrokenPipeError, ConnectionResetError):
            pass

    forward_task = asyncio.create_task(_forward_camera()) if camera_queue else None

    try:
        async for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
    finally:
        if forward_task:
            forward_task.cancel()
            try:
                await forward_task
            except asyncio.CancelledError:
                pass
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()


def _describe_action_space(space) -> dict:
    """Return a JSON-friendly description of the action space."""
    import gymnasium.spaces as spaces

    if isinstance(space, spaces.Discrete):
        return {"type": "discrete", "n": int(space.n)}
    elif isinstance(space, spaces.Box):
        return {
            "type": "box",
            "shape": list(space.shape),
            "low": space.low.tolist(),
            "high": space.high.tolist(),
        }
    elif isinstance(space, spaces.MultiDiscrete):
        return {"type": "multi_discrete", "nvec": space.nvec.tolist()}
    return {"type": str(type(space).__name__)}


def _render_frame(env) -> str:
    """Render a frame and return base64 PNG."""
    try:
        raw = env.render()
        if isinstance(raw, tuple):
            raw = raw[0]
        if raw is None:
            return ""
        arr = np.asarray(raw)
        if hasattr(arr, "cpu"):
            arr = arr.cpu().numpy()
        if arr.ndim < 2:
            return ""
        from PIL import Image

        img = Image.fromarray(arr.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        import traceback
        traceback.print_exc()
        return ""


async def _ws_reader(websocket, action_queue: asyncio.Queue, camera_queue: asyncio.Queue | None = None) -> None:
    """Background task that reads from the websocket and dispatches commands."""
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            if "action" in data:
                await action_queue.put(data["action"])
            elif data.get("cmd") == "reset":
                await action_queue.put("reset")
            elif "camera" in data and camera_queue is not None:
                await camera_queue.put(data)
    except Exception:
        pass


async def stream_env_frames(
    env_id: str,
    max_steps: int = 2000,
    websocket=None,
) -> AsyncIterator[dict]:
    """Load an environment, yield frame dicts. Accepts actions via websocket.

    Genesis envs run in a subprocess with camera commands piped via stdin.
    Pygame envs run in-process with interactive action control.
    """
    env_dir = ENVS_DIR / env_id
    env_py = env_dir / "env.py"
    if not env_py.exists():
        yield {"error": f"No env.py in {env_id}"}
        return

    is_genesis = _is_genesis_env(env_dir)

    if is_genesis:
        camera_queue: asyncio.Queue | None = None
        reader_task: asyncio.Task | None = None

        if websocket is not None:
            camera_queue = asyncio.Queue()
            action_queue_dummy: asyncio.Queue = asyncio.Queue()
            reader_task = asyncio.create_task(
                _ws_reader(websocket, action_queue_dummy, camera_queue)
            )

        try:
            async for frame_data in _stream_via_subprocess(env_dir, max_steps, camera_queue):
                yield frame_data
        finally:
            if reader_task:
                reader_task.cancel()
                try:
                    await reader_task
                except asyncio.CancelledError:
                    pass
        return

    try:
        env_cls = _load_env_class(env_dir)
    except Exception as e:
        yield {"error": f"Failed to load env: {e}"}
        return

    if env_cls is None:
        yield {"error": "No gymnasium.Env subclass found in env.py"}
        return

    action_queue: asyncio.Queue = asyncio.Queue()
    reader_task: asyncio.Task | None = None
    if websocket is not None:
        reader_task = asyncio.create_task(_ws_reader(websocket, action_queue))

    env = env_cls(render_mode="rgb_array")
    try:
        obs, info = env.reset()
        space_desc = _describe_action_space(env.action_space)

        init_frame = _render_frame(env)
        yield {
            "type": "init",
            "action_space": space_desc,
            "step": 0,
            "frame": init_frame,
            "reward": 0.0,
            "done": False,
            "actions": [],
            "info": {},
        }

        for step in range(max_steps):
            client_action = None
            try:
                client_action = action_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

            if client_action == "reset":
                obs, info = env.reset()
                yield {
                    "step": step,
                    "frame": _render_frame(env),
                    "reward": 0.0,
                    "done": False,
                    "actions": [],
                    "info": {},
                }
                continue

            if client_action is not None:
                import gymnasium.spaces as spaces

                if isinstance(env.action_space, spaces.Discrete):
                    action = int(client_action) if not isinstance(client_action, list) else int(client_action[0])
                else:
                    action = np.array(client_action, dtype=np.float32)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            frame_b64 = _render_frame(env)
            yield {
                "step": step,
                "frame": frame_b64,
                "reward": float(reward),
                "done": bool(terminated or truncated),
                "actions": action.tolist() if hasattr(action, "tolist") else [int(action)],
                "info": {
                    k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                    for k, v in (info or {}).items()
                    if isinstance(v, (int, float, str, np.floating))
                },
            }

            if terminated or truncated:
                obs, info = env.reset()

            await asyncio.sleep(0.03)
    finally:
        if reader_task is not None:
            reader_task.cancel()
            try:
                await reader_task
            except asyncio.CancelledError:
                pass
        env.close()
