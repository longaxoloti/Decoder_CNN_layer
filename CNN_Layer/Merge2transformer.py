import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

class MergeToTransformerInput(nn.Module):
    """
    Combine pooled vectors and passthrough tokens into a final sequence.
    Options: interleave or concat-last. Also optionally add positional embeddings for passthrough.
    """
    def __init__(self, D: int, Dmodel: int, interleave: bool = False):
        super().__init__()
        self.interleave = interleave
        self.to_model = nn.Linear(D, Dmodel) if D != Dmodel else nn.Identity()
        self.ln = nn.LayerNorm(Dmodel)

    def forward(self, pooled: torch.Tensor, passthrough: torch.Tensor, passthrough_pos: Optional[torch.Tensor] = None):
        # pooled: (B, M, D) ; passthrough: (B, K, D)
        B, M, D = pooled.shape
        K = passthrough.shape[1]
        pooled_mapped = self.to_model(pooled)  # (B, M, Dmodel)
        pt_mapped = self.to_model(passthrough)  # (B, K, Dmodel)
        if self.interleave:
            # naive interleave: keep relative order unknown; user must supply order
            # produce sequence length M+K
            seq = torch.cat([pooled_mapped, pt_mapped], dim=1)
        else:
            seq = torch.cat([pt_mapped, pooled_mapped], dim=1)  # (B, K+M, Dmodel)
        seq = self.ln(seq)
        return seq  # ready to feed into transformer (B, L_t, Dmodel)