"""Test individual tools and the full orchestrator pipeline."""

from __future__ import annotations

import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import ROOT_DIR, ENVS_DIR, ASSETS_DIR
from core.orchestrator import AgentEvent, create_orchestrator


def on_event(event: AgentEvent) -> None:
    """Print events as they happen."""
    t = event.type
    d = event.data
    if t == "tool_start":
        print(f"  -> {d['tool']}({list(d['args'].keys())})")
    elif t == "tool_end":
        print(f"  <- {d['tool']}: {d['result'][:120]}")
    elif t == "validation":
        print(f"  [VAL] {d}")
    elif t == "error":
        print(f"  [ERR] {d['message']}")
    elif t == "message":
        print(f"  [MSG] {d['content'][:200]}")


def test_coding_tools() -> None:
    """Test basic file operations."""
    from core.tools.coding import file_write, file_read, file_edit, dir_list, shell

    print("\n=== Testing coding tools ===")

    r = file_write.invoke({"path": "envs/_test_tmp/hello.txt", "contents": "hello\nworld"})
    assert "Wrote" in r, f"file_write failed: {r}"
    print(f"  file_write: {r}")

    r = file_read.invoke({"path": "envs/_test_tmp/hello.txt"})
    assert "hello" in r, f"file_read failed: {r}"
    print(f"  file_read: OK")

    r = file_edit.invoke({"path": "envs/_test_tmp/hello.txt", "old_string": "hello", "new_string": "HELLO"})
    assert "Edited" in r, f"file_edit failed: {r}"
    print(f"  file_edit: {r}")

    r = dir_list.invoke({"path": "core", "max_depth": 1})
    assert "tools" in r, f"dir_list failed: {r}"
    print(f"  dir_list: OK")

    r = shell.invoke({"command": "echo test_ok"})
    assert "test_ok" in r, f"shell failed: {r}"
    print(f"  shell: OK")

    shutil.rmtree(ENVS_DIR / "_test_tmp", ignore_errors=True)
    print("  All coding tools PASS")


def test_eval_tool() -> None:
    """Test eval pipeline with a known-good minimal env."""
    from core.tools.eval_tool import eval_env

    print("\n=== Testing eval tool ===")
    env_dir = ENVS_DIR / "_test_eval"
    env_dir.mkdir(parents=True, exist_ok=True)

    (env_dir / "env.py").write_text('''\
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class TestEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.observation_space = spaces.Box(0, 4, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.render_mode = render_mode
        self._pos = np.zeros(2, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = np.zeros(2, dtype=np.float32)
        return self._pos.copy(), {}

    def step(self, action):
        self._pos = np.clip(self._pos + 0.1, 0, 4)
        return self._pos.copy(), -0.01, False, False, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return np.zeros((64, 64, 3), dtype=np.uint8)
        return None

    def close(self):
        pass
''', encoding="utf-8")

    r = eval_env.invoke({"env_id": "_test_eval"})
    print(f"  {r}")
    assert "[FAIL]" not in r, f"eval_tool failed on valid env: {r}"
    print("  Eval tool PASS")

    shutil.rmtree(env_dir, ignore_errors=True)


def test_orchestrator_gym(prompt: str | None = None) -> None:
    """Test full orchestrator: generate a Gym env from text."""
    print("\n=== Testing orchestrator (Gym) ===")

    prompt = prompt or (
        "Create a simple 2D 'maze_runner' environment. "
        "5x5 grid with walls (hardcoded L-shape wall). "
        "Agent starts top-left, goal bottom-right. "
        "Actions: up/down/left/right (Discrete 4). "
        "Obs: flat Box(2) with agent (row,col). "
        "Reward: +1 at goal, -0.01 per step, -0.1 hitting wall. "
        "Truncate at 100 steps. Render with Pygame rgb_array."
    )

    print(f"  Prompt: {prompt[:100]}...")
    agent = create_orchestrator(on_event=on_event)
    response = agent.run(prompt)
    print(f"\n  Response: {response[:300]}")

    env_dirs = list(ENVS_DIR.iterdir())
    generated = [d for d in env_dirs if d.is_dir() and (d / "env.py").exists()]
    print(f"\n  Generated envs: {[d.name for d in generated]}")
    assert len(generated) > 0, "No environment was generated"
    print("  Orchestrator (Gym) PASS")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["tools", "eval", "gym", "all"], default="all")
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()

    if args.test in ("tools", "all"):
        test_coding_tools()
    if args.test in ("eval", "all"):
        test_eval_tool()
    if args.test in ("gym", "all"):
        test_orchestrator_gym(args.prompt)

    print("\n" + "=" * 60)
    print("All requested tests completed!")
