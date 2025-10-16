import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from utils import topk_from_scores

class TopKSelector(nn.Module):
    """
    Compute per-position scores and select top-K positions.
    score_fn: small MLP(LN(feat)) -> scalar
    Returns indices and gated embeddings (mix original & pooled if requested)
    """
    def __init__(self, D: int, hidden: int = 256, scorer_bias: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(D)
        self.scorer = nn.Sequential(
            nn.Linear(D, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )
        self.scorer_bias = scorer_bias

    def forward(self, features: torch.Tensor, k: int):
        # features: (B, L, D)
        B, L, D = features.shape
        if L == 0:
            return features.new_empty((B, 0, D)), features.new_empty((B, L)), torch.empty((B, 0), dtype = torch.long, device = features.device)
        x = self.ln(features)  # (B, L, D)
        scores = self.scorer(x).squeeze(-1) + self.scorer_bias  # (B, L)
        # raw scores for topk
        idx = topk_from_scores(scores, k=k)  # (B, k) or (k, )
        if idx.dim() == 1:
            selected = features[:, idx, :]  # (B, k, D)
        else:
            # idx: (B, k)
            batch_idx = torch.arange(B, device=features.device).unsqueeze(-1)
            selected = features[batch_idx, idx, :]  # (B, k, D)
        return selected, scores, idx