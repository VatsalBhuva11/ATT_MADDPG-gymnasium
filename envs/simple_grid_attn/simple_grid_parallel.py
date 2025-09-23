from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional

from pettingzoo.utils import ParallelEnv
from gymnasium import spaces
import pygame


# Simple cooperative multi-agent gridworld:
# - Agents try to reach their own goals; shaping reward for moving closer, +10 on goal, -0.01 step cost
# - Episode ends when all agents reach goals or max_steps
# - Observations: own position, goal position, other agents relative positions (clipped), grid size
# - Discrete actions: 5 (stay, up, down, left, right)
# - Adjustable number of agents, grid size, max steps, seed


action_meanings = ["stay", "up", "down", "left", "right"]


def simple_grid_v0(
    num_agents: int = 3,
    grid_size: int = 7,
    max_steps: int = 100,
    random_spawn: bool = True,
    seed: Optional[int] = None,
):
    return SimpleGridParallelEnv(
        num_agents=num_agents,
        grid_size=grid_size,
        max_steps=max_steps,
        random_spawn=random_spawn,
        seed=seed,
    )


class SimpleGridParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "simple_grid_attn_v0"}

    def __init__(
        self,
        num_agents: int = 3,
        grid_size: int = 7,
        max_steps: int = 100,
        random_spawn: bool = True,
        seed: Optional[int] = None,
    ):
        assert num_agents >= 2
        assert grid_size >= 3
        self.n_agents = num_agents
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.random_spawn = random_spawn
        self._step_count = 0

        self.pos = None  # shape (N, 2)
        self.goals = None  # shape (N, 2)
        self._rng = np.random.RandomState(seed)
        self._agent_ids = [f"agent_{i}" for i in range(num_agents)]
        self.possible_agents = list(self._agent_ids)

        # action/observation spaces
        self.action_space = spaces.Discrete(5)
        # obs: [own_x, own_y, goal_x, goal_y, other_rel (N-1)*2, grid]
        self.max_other = self.n_agents - 1
        obs_dim = 4 + self.max_other * 2 + 1
        low = np.array([0, 0, 0, 0] + [-grid_size] * (self.max_other * 2) + [grid_size], dtype=np.int32)
        high = np.array([grid_size - 1, grid_size - 1, grid_size - 1, grid_size - 1] + [grid_size] * (self.max_other * 2) + [grid_size], dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        self.render_mode = None
        self._surface = None

    def observation_spaces(self):
        return {aid: self.observation_space for aid in self.possible_agents}

    def action_spaces(self):
        return {aid: self.action_space for aid in self.possible_agents}

    @property
    def agents(self):
        return list(self._agent_ids)

    def seed(self, seed: Optional[int] = None):
        self._rng.seed(seed)

    def _sample_empty_positions(self, k: int) -> np.ndarray:
        coords = set()
        output = []
        while len(output) < k:
            xy = (int(self._rng.randint(0, self.grid_size)), int(self._rng.randint(0, self.grid_size)))
            if xy not in coords:
                coords.add(xy)
                output.append(xy)
        return np.array(output, dtype=np.int32)

    def _reset_positions(self):
        if self.random_spawn:
            all_positions = self._sample_empty_positions(self.n_agents * 2)
            self.pos = all_positions[: self.n_agents]
            self.goals = all_positions[self.n_agents : self.n_agents * 2]
        else:
            xs = np.linspace(0, self.grid_size - 1, self.n_agents, dtype=int)
            self.pos = np.stack([xs, np.zeros_like(xs)], axis=1)
            self.goals = np.stack([xs[::-1], np.full_like(xs, self.grid_size - 1)], axis=1)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed(seed)
        self._step_count = 0
        self._reset_positions()
        obs = self._get_all_obs()
        infos = {aid: {} for aid in self.agents}
        return obs, infos

    def _get_obs(self, idx: int) -> np.ndarray:
        own = self.pos[idx]
        goal = self.goals[idx]
        rels = []
        for j in range(self.n_agents):
            if j == idx:
                continue
            rel = self.pos[j] - own
            rel = np.clip(rel, -self.grid_size, self.grid_size)
            rels.extend(list(rel))
        # pad if needed
        while len(rels) < (self.max_other * 2):
            rels.extend([0, 0])
        vec = np.array([own[0], own[1], goal[0], goal[1]] + rels + [self.grid_size], dtype=np.int32)
        return vec

    def _get_all_obs(self) -> Dict[str, np.ndarray]:
        return {aid: self._get_obs(i) for i, aid in enumerate(self.agents)}

    def step(self, actions: Dict[str, int]):
        self._step_count += 1
        # apply actions
        for i, aid in enumerate(self.agents):
            a = int(actions.get(aid, 0))
            dx, dy = 0, 0
            if a == 1:
                dy = -1
            elif a == 2:
                dy = 1
            elif a == 3:
                dx = -1
            elif a == 4:
                dx = 1
            newx = int(np.clip(self.pos[i, 0] + dx, 0, self.grid_size - 1))
            newy = int(np.clip(self.pos[i, 1] + dy, 0, self.grid_size - 1))
            self.pos[i] = (newx, newy)

        # compute rewards
        rewards = {}
        terminated = {}
        truncated = {}
        infos = {aid: {"action_meaning": action_meanings[int(actions.get(aid, 0))]} for aid in self.agents}

        all_done = True
        for i, aid in enumerate(self.agents):
            prev_dist = np.abs(self.pos[i] - self.goals[i]).sum()
            # reward shaping: closer to goal positive
            step_reward = -0.01
            if np.array_equal(self.pos[i], self.goals[i]):
                step_reward += 10.0
            else:
                all_done = False
                # encourage movement toward goal
                # approximate improvement by hypothetically moving closer (already moved), use negative manhattan distance as potential
                step_reward += -0.1 * (prev_dist)
            rewards[aid] = float(step_reward)
            terminated[aid] = bool(np.array_equal(self.pos[i], self.goals[i]))
            truncated[aid] = False

        if self._step_count >= self.max_steps:
            for aid in self.agents:
                truncated[aid] = True

        obs = self._get_all_obs()
        return obs, rewards, terminated, truncated, infos

    def render(self, mode: str = "human"):
        cell = 50
        margin = 2
        width = self.grid_size * cell
        height = self.grid_size * cell
        if self._surface is None:
            pygame.init()
            if mode == "human":
                self._surface = pygame.display.set_mode((width, height))
            else:
                self._surface = pygame.Surface((width, height))
        surf = self._surface
        surf.fill((30, 30, 30))
        # grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(x * cell + margin, y * cell + margin, cell - 2 * margin, cell - 2 * margin)
                pygame.draw.rect(surf, (50, 50, 50), rect, 1)
        # goals
        for i in range(self.n_agents):
            gx, gy = self.goals[i]
            rect = pygame.Rect(gx * cell + margin, gy * cell + margin, cell - 2 * margin, cell - 2 * margin)
            pygame.draw.rect(surf, (60, 120, 60), rect)
        # agents
        colors = [(200, 80, 80), (80, 160, 220), (220, 200, 80), (140, 80, 200), (80, 200, 160)]
        for i in range(self.n_agents):
            ax, ay = self.pos[i]
            rect = pygame.Rect(ax * cell + margin + 6, ay * cell + margin + 6, cell - 2 * margin - 12, cell - 2 * margin - 12)
            pygame.draw.rect(surf, colors[i % len(colors)], rect)
        if mode == "human":
            pygame.display.flip()
        else:
            arr = pygame.surfarray.array3d(surf)
            return np.transpose(arr, (1, 0, 2))

    def close(self):
        if self._surface is not None:
            pygame.quit()
            self._surface = None 