import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

class AttentionPooling(nn.Module):
    """
    Use multi-head cross-attention where queries are learnable (M, D).
    Returns pooled vectors of shape (B, M, D).
    Implement via nn.MultiheadAttention (needs sequence-first) or custom matmul for clarity.
    """
    def __init__(self, D: int, M: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.D = D
        self.M = M
        self.n_heads = n_heads
        self.pool_queries = nn.Parameter(torch.randn(M, D) * (D ** -0.5))
        # use linear maps for Q/K/V
        self.q_proj = nn.Linear(D, D)
        self.k_proj = nn.Linear(D, D)
        self.v_proj = nn.Linear(D, D)
        self.out_proj = nn.Linear(D, D)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, conv_feats: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # conv_feats: (B, L', D)
        B, Lp, D = conv_feats.shape
        # expand queries -> (B, M, D)
        q = self.pool_queries.unsqueeze(0).expand(B, -1, -1)  # (B, M, D)
        Q = self.q_proj(q)  # (B, M, D)
        K = self.k_proj(conv_feats)  # (B, L', D)
        V = self.v_proj(conv_feats)  # (B, L', D)

        def reshape_for_heads(x):
            # x: (B, S, D) -> (B, n_heads, S, D_head)
            d_head = D // self.n_heads
            x = x.view(B, -1, self.n_heads, d_head).transpose(1,2)  # (B, n_heads, S, d_head)
            return x

        Qh = reshape_for_heads(Q)  # (B, n_heads, M, d_head)
        Kh = reshape_for_heads(K)  # (B, n_heads, L', d_head)
        Vh = reshape_for_heads(V)  # (B, n_heads, L', d_head)
        scale = (D // self.n_heads) ** -0.5
        attn_logits = torch.einsum('bhmd,bhnd->bhmn', Qh, Kh) * scale  # (B, n_heads, M, L')
        if mask is not None:
            # mask (B, L') -> expand
            attn_logits = attn_logits.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn = torch.softmax(attn_logits, dim=-1)  # (B, n_heads, M, L')
        attn = self.attn_dropout(attn)
        out_h = torch.einsum('bhmn,bhnd->bhmd', attn, Vh)  # (B, n_heads, M, d_head)
        out = out_h.transpose(1,2).contiguous().view(B, self.M, D)  # (B, M, D)
        out = self.out_proj(out)  # (B, M, D)
        return out