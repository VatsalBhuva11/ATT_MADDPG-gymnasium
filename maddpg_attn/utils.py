from __future__ import annotations
import torch
import numpy as np


def one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    out = torch.zeros(indices.size(0), num_classes, device=indices.device)
    out.scatter_(1, indices.view(-1, 1), 1.0)
    return out


def gumbel_softmax(logits: torch.Tensor, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
    gumbels = -torch.empty_like(logits).exponential_().log()
    y = torch.softmax((logits + gumbels) / tau, dim=-1)
    if hard:
        index = y.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(y).scatter_(-1, index, 1.0)
        y = (y_hard - y).detach() + y
    return y


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) 