import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Any, Optional, Tuple

class CollectorWorldEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode: Optional[str] = None, grid_size: int = 6, max_steps: int = 100) -> None:
        '''Create the collector world environment.'''
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata['render_modes']:
            raise ValueError(f'Unsupported render_mode {render_mode!r}.')
        if grid_size < 5:
            raise ValueError('grid_size must be at least 5 to place all items.')
        self.render_mode = render_mode
        self.grid_size = grid_size
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)

        self.agent_start = np.array((0, 0), dtype=np.int64)
        self.item_positions = [
            (grid_size - 1, 0),
            (grid_size - 1, grid_size - 1),
            (0, grid_size - 1),
            (grid_size // 2, 1),
            (2, grid_size - 2),
        ]

        self.collected: list[bool] = [False for _ in self.item_positions]
        self.agent_pos = self.agent_start.copy()
        self.steps = 0

        self.cell_size = 64
        self.window_size = self.grid_size * self.cell_size

        self.canvas: Optional[pygame.Surface] = None
        self.window: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None

        self.background_color = (245, 240, 230)
        self.agent_color = (65, 92, 87)
        self.item_color = (103, 138, 99)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> Tuple[np.ndarray, dict[str, Any]]:
        '''Reset the environment to the initial state.'''
        super().reset(seed=seed)
        self.steps = 0
        self.agent_pos = self.agent_start.copy()
        self.collected = [False for _ in self.item_positions]
        observation = self._get_observation()
        info: dict[str, Any] = {'items_remaining': self._items_remaining()}
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        '''Move the agent and update the collection state.'''
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action {action}.')
        self._move_agent(action)
        reward = -0.05
        reward += self._update_collection()
        self.steps += 1

        terminated = self._items_remaining() == 0
        truncated = self.steps >= self.max_steps

        observation = self._get_observation()
        info: dict[str, Any] = {
            'items_remaining': self._items_remaining(),
            'items_collected': int(sum(self.collected)),
        }
        return observation, float(reward), terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        '''Render the environment in the configured mode.'''
        if self.render_mode is None:
            raise ValueError('render_mode is None; set render_mode when constructing the environment.')
        self._ensure_render_initialized()
        self._draw_scene()

        if self.render_mode == 'human':
            assert self.window is not None
            assert self.canvas is not None
            self.window.blit(self.canvas, (0, 0))
            pygame.display.flip()
            assert self.clock is not None
            self.clock.tick(self.metadata['render_fps'])
            return None

        assert self.canvas is not None
        frame = pygame.surfarray.array3d(self.canvas)
        return np.transpose(frame, (1, 0, 2))

    def close(self) -> None:
        '''Shut down the renderer and clean up resources.'''
        if self.window is not None:
            pygame.display.quit()
            self.window = None
        if self.canvas is not None:
            self.canvas = None
        if self.clock is not None:
            self.clock = None
        pygame.quit()

    def _ensure_render_initialized(self) -> None:
        if self.canvas is None:
            pygame.init()
            self.canvas = pygame.Surface((self.window_size, self.window_size))
        if self.render_mode == 'human' and self.window is None:
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption('Collector World')
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.render_mode == 'human':
            pygame.event.pump()

    def _draw_scene(self) -> None:
        assert self.canvas is not None
        self.canvas.fill(self.background_color)
        for index, item_pos in enumerate(self.item_positions):
            if not self.collected[index]:
                self._draw_item(item_pos)
        self._draw_agent()

    def _draw_item(self, position: Tuple[int, int]) -> None:
        assert self.canvas is not None
        padding = self.cell_size // 4
        rect = pygame.Rect(
            position[0] * self.cell_size + padding,
            position[1] * self.cell_size + padding,
            self.cell_size - 2 * padding,
            self.cell_size - 2 * padding,
        )
        pygame.draw.rect(self.canvas, self.item_color, rect, border_radius=6)

    def _draw_agent(self) -> None:
        assert self.canvas is not None
        padding = self.cell_size // 5
        rect = pygame.Rect(
            int(self.agent_pos[0]) * self.cell_size + padding,
            int(self.agent_pos[1]) * self.cell_size + padding,
            self.cell_size - 2 * padding,
            self.cell_size - 2 * padding,
        )
        pygame.draw.rect(self.canvas, self.agent_color, rect, border_radius=8)

    def _move_agent(self, action: int) -> None:
        deltas = {
            0: (0, -1),
            1: (1, 0),
            2: (0, 1),
            3: (-1, 0),
        }
        dx, dy = deltas[action]
        new_x = int(np.clip(self.agent_pos[0] + dx, 0, self.grid_size - 1))
        new_y = int(np.clip(self.agent_pos[1] + dy, 0, self.grid_size - 1))
        self.agent_pos = np.array((new_x, new_y), dtype=np.int64)

    def _update_collection(self) -> float:
        reward = 0.0
        pos = (int(self.agent_pos[0]), int(self.agent_pos[1]))
        for index, item_pos in enumerate(self.item_positions):
            if not self.collected[index] and pos == item_pos:
                self.collected[index] = True
                reward += 1.0
        return reward

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        denom = float(self.grid_size - 1)
        obs[0] = self.agent_pos[0] / denom
        obs[1] = self.agent_pos[1] / denom
        for index, collected in enumerate(self.collected):
            obs[2 + index] = 1.0 if collected else 0.0
        return obs

    def _items_remaining(self) -> int:
        return int(sum(1 for collected in self.collected if not collected))
