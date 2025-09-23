from __future__ import annotations
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .models import Actor, AttentionCritic
from .utils import one_hot, gumbel_softmax


def to_device(x, device):
    if isinstance(x, list):
        return [to_device(t, device) for t in x]
    return x.to(device)


class MADDPG(nn.Module):
    def __init__(
        self,
        obs_dims: List[int],
        act_dims: List[int],
        actor_hidden: List[int] = [128, 128],
        critic_embed: int = 128,
        critic_heads: int = 2,
        critic_hidden: List[int] = [128, 128],
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.95,
        tau: float = 0.01,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_agents = len(obs_dims)
        self.obs_dims = obs_dims
        self.act_dims = act_dims
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau

        self.actors = nn.ModuleList([Actor(obs_dims[i], act_dims[i], actor_hidden) for i in range(self.num_agents)])
        self.target_actors = nn.ModuleList([Actor(obs_dims[i], act_dims[i], actor_hidden) for i in range(self.num_agents)])
        self.critic = AttentionCritic(obs_dims, act_dims, embed_dim=critic_embed, num_heads=critic_heads, hidden_dims=critic_hidden)
        self.target_critic = AttentionCritic(obs_dims, act_dims, embed_dim=critic_embed, num_heads=critic_heads, hidden_dims=critic_hidden)

        self.actors.to(self.device)
        self.target_actors.to(self.device)
        self.critic.to(self.device)
        self.target_critic.to(self.device)

        self.actor_opt = [optim.Adam(self.actors[i].parameters(), lr=actor_lr) for i in range(self.num_agents)]
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self._hard_update_targets()

    @torch.no_grad()
    def _hard_update_targets(self):
        for i in range(self.num_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    @torch.no_grad()
    def _soft_update(self):
        for i in range(self.num_agents):
            for p, tp in zip(self.actors[i].parameters(), self.target_actors[i].parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    @torch.no_grad()
    def act(self, obs_dict: Dict[str, np.ndarray], agent_order: List[str], explore: bool = True) -> Dict[str, int]:
        actions = {}
        for i, aid in enumerate(agent_order):
            obs = torch.tensor(obs_dict[aid], dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.actors[i](obs)
            if explore:
                probs = torch.softmax(logits, dim=-1)
                a = torch.distributions.Categorical(probs=probs).sample().item()
            else:
                a = torch.argmax(logits, dim=-1).item()
            actions[aid] = int(a)
        return actions

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        # batch fields are already tensors on device
        B = batch['obs_0'].size(0)
        obs_list = [batch[f'obs_{i}'] for i in range(self.num_agents)]
        act_idx_list = [batch[f'act_{i}'] for i in range(self.num_agents)]  # [B]
        act_oh_list = [one_hot(act_idx_list[i], self.act_dims[i]) for i in range(self.num_agents)]
        rew_list = [batch[f'rew_{i}'].unsqueeze(1) for i in range(self.num_agents)]
        next_obs_list = [batch[f'next_obs_{i}'] for i in range(self.num_agents)]
        done_list = [batch[f'done_{i}'].unsqueeze(1) for i in range(self.num_agents)]

        # Critic loss
        with torch.no_grad():
            next_logits = [self.target_actors[i](next_obs_list[i]) for i in range(self.num_agents)]
            next_actions = [gumbel_softmax(next_logits[i], hard=True) for i in range(self.num_agents)]  # [B, Ai]
            target_q_all = self.target_critic(next_obs_list, next_actions)  # [B, N]
            # y_i = r_i + gamma * (1-done_i) * target_q_i
            y = []
            for i in range(self.num_agents):
                y.append(rew_list[i] + (1.0 - done_list[i]) * self.gamma * target_q_all[:, i : i + 1])
            y = torch.cat(y, dim=1)  # [B, N]
        current_q_all = self.critic(obs_list, act_oh_list)
        critic_loss = nn.MSELoss()(current_q_all, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # Actor loss (maximize Q for own action while others fixed by current policy)
        actor_losses = []
        for i in range(self.num_agents):
            logits_i = self.actors[i](obs_list[i])
            act_i = gumbel_softmax(logits_i, hard=True)  # differentiable one-hot
            # others use their current sampled actions from batch as baseline
            act_for_critic = [act_oh_list[j].detach() if j != i else act_i for j in range(self.num_agents)]
            q_all = self.critic(obs_list, act_for_critic)
            # maximize q_i -> minimize -q_i
            actor_loss = (-q_all[:, i]).mean()
            self.actor_opt[i].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[i].parameters(), 1.0)
            self.actor_opt[i].step()
            actor_losses.append(actor_loss.detach())

        self._soft_update()
        logs = {"critic_loss": float(critic_loss.detach().cpu().item())}
        for i, al in enumerate(actor_losses):
            logs[f"actor_{i}_loss"] = float(al.cpu().item())
        return logs

    def save(self, path: str):
        state = {
            "actors": [a.state_dict() for a in self.actors],
            "critic": self.critic.state_dict(),
        }
        torch.save(state, path)

    def load(self, path: str, map_location: str | None = None):
        state = torch.load(path, map_location=map_location or self.device)
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(state["actors"][i])
        self.critic.load_state_dict(state["critic"]) 