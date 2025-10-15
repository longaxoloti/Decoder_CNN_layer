import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

def topk_from_scores(scores: torch.Tensor, k: int):
    """
    Safe top-k selector.
    - If k > L: set k = L.
    Returns:
      - if scores.dim()==1: tensor shape (k,)
      - if scores.dim()==2: tensor shape (B, k)
    """
    if k <= 0:
        if scores.dim() == 1:
            return torch.empty(0, dtype=torch.long, device=scores.device)
        elif scores.dim() == 2:
            B = scores.size(0)
            return torch.empty((B, 0), dtype=torch.long, device=scores.device)
        else:
            raise ValueError("scores.dim must be 1 or 2")

    if scores.dim() == 1:
        L = scores.size(0)
        k = min(k, L)
        if k == 0:
            return torch.empty(0, dtype=torch.long, device=scores.device)
        vals, idx = torch.topk(scores, k)
        return idx
    elif scores.dim() == 2:
        L = scores.size(1)
        k = min(k, L)
        if k == 0:
            B = scores.size(0)
            return torch.empty((B, 0), dtype=torch.long, device=scores.device)
        vals, idx = torch.topk(scores, k, dim=1)
        return idx
    else:
        raise ValueError("scores.dim must be 1 or 2")