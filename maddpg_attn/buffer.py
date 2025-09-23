from __future__ import annotations
from typing import List, Dict
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, num_agents: int, obs_dims: List[int], capacity: int = 100000, device: str = "cpu"):
        self.num_agents = num_agents
        self.obs_dims = obs_dims
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        self.device = torch.device(device)

        self.obs = [np.zeros((capacity, obs_dims[i]), dtype=np.float32) for i in range(num_agents)]
        self.act = [np.zeros((capacity,), dtype=np.int64) for _ in range(num_agents)]
        self.rew = [np.zeros((capacity,), dtype=np.float32) for _ in range(num_agents)]
        self.next_obs = [np.zeros((capacity, obs_dims[i]), dtype=np.float32) for i in range(num_agents)]
        self.done = [np.zeros((capacity,), dtype=np.float32) for _ in range(num_agents)]

    def add(self, obs_list: List[np.ndarray], act_list: List[int], rew_list: List[float], next_obs_list: List[np.ndarray], done_list: List[bool]):
        idx = self.ptr
        for i in range(self.num_agents):
            self.obs[i][idx] = obs_list[i].astype(np.float32)
            self.act[i][idx] = int(act_list[i])
            self.rew[i][idx] = float(rew_list[i])
            self.next_obs[i][idx] = next_obs_list[i].astype(np.float32)
            self.done[i][idx] = float(done_list[i])
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = {}
        for i in range(self.num_agents):
            batch[f'obs_{i}'] = torch.tensor(self.obs[i][idxs], dtype=torch.float32, device=self.device)
            batch[f'act_{i}'] = torch.tensor(self.act[i][idxs], dtype=torch.int64, device=self.device)
            batch[f'rew_{i}'] = torch.tensor(self.rew[i][idxs], dtype=torch.float32, device=self.device)
            batch[f'next_obs_{i}'] = torch.tensor(self.next_obs[i][idxs], dtype=torch.float32, device=self.device)
            batch[f'done_{i}'] = torch.tensor(self.done[i][idxs], dtype=torch.float32, device=self.device)
        return batch 