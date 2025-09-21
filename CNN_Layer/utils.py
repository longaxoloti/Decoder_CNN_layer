import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

def topk_from_scores(scores: torch.Tensor, k: int):
    # scores: (L,) or (B, L) -> returns indices sorted descending
    if scores.dim() == 1:
        vals, idx = torch.topk(scores, k)
        return idx
    elif scores.dim() == 2:
        vals, idx = torch.topk(scores, k, dim=1)
        return idx
    else:
        raise ValueError("scores.dim must be 1 or 2")