import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

class ReconstructionHead(nn.Module):
    """
    Map compressed/pooling representation back to per-token logits or embeddings.
    We provide a cross-attention decoder approach: queries = original positions (learnable or embedding),
    keys/values = compressed representation (pooled + passthrough).
    Simpler: transposed conv upsample from L' -> L (not implemented here).
    """
    def __init__(self, D: int, Dmodel: int, n_heads: int = 8):
        super().__init__()
        self.q_proj = nn.Linear(D, Dmodel)
        self.k_proj = nn.Linear(Dmodel, Dmodel)
        self.v_proj = nn.Linear(Dmodel, Dmodel)
        self.out = nn.Linear(Dmodel, D)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor):
        # queries: (B, L_query, D) -> often original positions or masked queries
        # memory: (B, L_mem, Dmodel)
        B, Lq, Dq = queries.shape
        q = self.q_proj(queries)  # (B, Lq, Dmodel)
        k = self.k_proj(memory)
        v = self.v_proj(memory)
        # scaled dot product (single head for simplicity) - can extend to multi-head
        d_head = q.size(-1)
        attn_logits = torch.einsum('bqd,bkd->bqk', q, k) / (d_head ** 0.5)
        attn = torch.softmax(attn_logits, dim=-1)
        out = torch.einsum('bqk,bkd->bqd', attn, v)
        out = self.out(out)  # (B, Lq, D)
        return out