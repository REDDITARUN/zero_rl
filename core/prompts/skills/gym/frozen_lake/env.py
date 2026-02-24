"""FrozenLake: grid navigation with holes and optional slippery ice.

Cross a frozen lake from Start to Goal without falling into Holes.
Demonstrates stochastic transitions, map-based level design,
and tabular transition probability computation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from .config import (
    ACTION_LABELS,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    FrozenLakeConfig,
)
from .renderer import FrozenLakeRenderer


class FrozenLakeEnv(gym.Env):
    """Frozen lake with configurable maps, slipperiness, and rewards.

    Observation:
        Discrete — single integer encoding the player's position
        (row * ncols + col).
    Actions:
        Discrete(4) — left / down / right / up.
    Rewards:
        Configurable: default +1 at goal, 0 at hole, 0 otherwise.
    Termination:
        Player reaches Goal ('G') or falls in Hole ('H').
    Truncation:
        Handled externally via TimeLimit wrapper.
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 4}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        config: Optional[FrozenLakeConfig] = None,
    ) -> None:
        self.cfg = config or FrozenLakeConfig()
        self.metadata["render_fps"] = self.cfg.render_fps

        desc = self.cfg.get_map()
        self.desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = self.desc.shape
        n_states = self.nrow * self.ncol

        self.observation_space = spaces.Discrete(n_states)
        self.action_space = spaces.Discrete(4)
        self.action_labels = ACTION_LABELS

        self._build_transitions()

        self.s: int = 0
        self.lastaction: Optional[int] = None
        self.render_mode = render_mode
        self._renderer: Optional[FrozenLakeRenderer] = None

    def _build_transitions(self) -> None:
        """Pre-compute transition table P[s][a] = [(prob, next_s, reward, done)]."""
        nS = self.nrow * self.ncol
        nA = 4
        self.P: Dict[int, Dict[int, list]] = {
            s: {a: [] for a in range(nA)} for s in range(nS)
        }

        fail_rate = (1.0 - self.cfg.success_rate) / 2.0

        def to_s(r: int, c: int) -> int:
            return r * self.ncol + c

        def inc(r: int, c: int, a: int) -> Tuple[int, int]:
            if a == LEFT:
                c = max(c - 1, 0)
            elif a == DOWN:
                r = min(r + 1, self.nrow - 1)
            elif a == RIGHT:
                c = min(c + 1, self.ncol - 1)
            elif a == UP:
                r = max(r - 1, 0)
            return r, c

        def transition(r: int, c: int, a: int) -> Tuple[int, float, bool]:
            nr, nc = inc(r, c, a)
            ns = to_s(nr, nc)
            letter = self.desc[nr, nc]
            done = bytes(letter) in b"GH"
            if letter == b"G":
                reward = self.cfg.goal_reward
            elif letter == b"H":
                reward = self.cfg.hole_reward
            else:
                reward = self.cfg.step_reward
            return ns, reward, done

        for row in range(self.nrow):
            for col in range(self.ncol):
                s = to_s(row, col)
                letter = self.desc[row, col]
                for a in range(nA):
                    li = self.P[s][a]
                    if letter in b"GH":
                        li.append((1.0, s, 0.0, True))
                    elif self.cfg.is_slippery:
                        for b in [(a - 1) % 4, a, (a + 1) % 4]:
                            prob = self.cfg.success_rate if b == a else fail_rate
                            li.append((prob, *transition(row, col, b)))
                    else:
                        li.append((1.0, *transition(row, col, a)))

        self.initial_state_distrib = np.array(
            self.desc == b"S"
        ).astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

    # -- core interface -------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        super().reset(seed=seed)
        probs = self.initial_state_distrib
        self.s = int(self.np_random.choice(len(probs), p=probs))
        self.lastaction = None
        if self.render_mode == "human":
            self.render()
        return self.s, {"prob": 1.0}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        transitions = self.P[self.s][action]
        probs = [t[0] for t in transitions]
        idx = int(self.np_random.choice(len(probs), p=probs))
        prob, new_s, reward, terminated = transitions[idx]

        self.s = int(new_s)
        self.lastaction = action

        if self.render_mode == "human":
            self.render()
        return self.s, float(reward), terminated, False, {"prob": prob}

    def render(self) -> Optional[np.ndarray | str]:
        if self.render_mode is None:
            return None
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self._renderer is None:
            self._renderer = FrozenLakeRenderer(self.cfg, self.desc, self.render_mode)
        return self._renderer.draw(self.s, self.lastaction)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # -- helpers --------------------------------------------------------------

    def _render_ansi(self) -> str:
        row, col = self.s // self.ncol, self.s % self.ncol
        desc = [[c.decode("utf-8") for c in line] for line in self.desc.tolist()]
        desc[row][col] = "\033[1;31m" + desc[row][col] + "\033[0m"
        header = ""
        if self.lastaction is not None:
            header = f"  ({ACTION_LABELS[self.lastaction]})\n"
        return header + "\n".join("".join(line) for line in desc) + "\n"
