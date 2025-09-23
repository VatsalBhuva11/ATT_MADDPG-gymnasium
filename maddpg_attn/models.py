from __future__ import annotations
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, activation: str = "relu", output_activation: str | None = None):
        super().__init__()
        layers: List[nn.Module] = []
        last = input_dim
        act = nn.ReLU if activation == "relu" else nn.Tanh
        for h in hidden_dims:
            layers += [nn.Linear(last, h), act()]
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)
        self.output_activation = output_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.output_activation == "tanh":
            return torch.tanh(y)
        return y


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        super().__init__()
        self.policy = MLP(obs_dim, hidden_dims, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.policy(obs)
        return logits


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        d_k = q.size(-1)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 2, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.attn = ScaledDotProductAttention(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, Tq, _ = q.shape
        Bk, Tk, _ = k.shape
        assert B == Bk
        # project
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        # reshape to heads
        def split_heads(x):
            B, T, C = x.shape
            x = x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            return x
        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)
        # scaled dot-product per head
        out = self.attn(q, k, v)  # [B, H, Tq, head_dim]
        # merge heads
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.embed_dim)
        out = self.out_proj(out)
        return out


class AttentionCritic(nn.Module):
    """
    Centralized critic: for each agent i, build query from (o_i, a_i) and keys/values from other agents' (o_j, a_j),
    compute attention, and produce a Q-value per agent. Outputs Q-values for all agents in batch.
    """

    def __init__(self, obs_dims: List[int], act_dims: List[int], embed_dim: int = 128, num_heads: int = 2, hidden_dims: List[int] = [128, 128]):
        super().__init__()
        assert len(obs_dims) == len(act_dims)
        self.num_agents = len(obs_dims)
        self.embed_dim = embed_dim
        self.obs_encoders = nn.ModuleList([MLP(obs_dims[i], [embed_dim], embed_dim) for i in range(self.num_agents)])
        self.act_encoders = nn.ModuleList([MLP(act_dims[i], [embed_dim], embed_dim) for i in range(self.num_agents)])
        self.mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.post = nn.ModuleList([MLP(embed_dim, hidden_dims, 1) for _ in range(self.num_agents)])

    def forward(self, obs_list: List[torch.Tensor], act_list: List[torch.Tensor]) -> torch.Tensor:
        # obs_list[i]: [B, obs_i], act_list[i]: [B, act_i] (one-hot for discrete)
        B = obs_list[0].size(0)
        # encode
        enc = []
        for i in range(self.num_agents):
            oi = self.obs_encoders[i](obs_list[i])
            ai = self.act_encoders[i](act_list[i])
            enc.append(torch.tanh(oi + ai))  # [B, E]
        enc = torch.stack(enc, dim=1)  # [B, N, E]

        q_values = []
        for i in range(self.num_agents):
            q = enc[:, i : i + 1, :]  # [B, 1, E]
            others_idx = [j for j in range(self.num_agents) if j != i]
            k = enc[:, others_idx, :]  # [B, N-1, E]
            v = k
            attn_out = self.mha(q, k, v)  # [B, 1, E]
            fused = attn_out.squeeze(1)  # [B, E]
            qi = self.post[i](fused)  # [B, 1]
            q_values.append(qi)
        q_values = torch.cat(q_values, dim=1)  # [B, N]
        return q_values 