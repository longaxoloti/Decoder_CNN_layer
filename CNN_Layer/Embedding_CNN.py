import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

class EmbeddingProjection(nn.Module):
    def __init__(self, vocab_size: Optional[int], D: int, Dproj: int,
                 use_embedding_table: bool = False):
        super().__init__()
        self.use_embedding_table = use_embedding_table
        if use_embedding_table:
            assert vocab_size is not None
            self.embed = nn.Embedding(vocab_size, D)
        else:
            self.embed = None
        self.D = D
        self.Dproj = Dproj
        if D != Dproj:
            self.proj = nn.Linear(D, Dproj)
        else:
            self.proj = nn.Identity()
        # positional: add rotary externally or we add simple learnable PE
        self.pos_emb = None  # keep external or add if desired

    def forward(self, tokens_or_embeds: torch.Tensor) -> torch.Tensor:
        # tokens_or_embeds: either (B, L) ints OR (B, L, D) float
        if self.use_embedding_table:
            x = self.embed(tokens_or_embeds)  # (B, L, D)
        else:
            x = tokens_or_embeds
        # project
        x_proj = self.proj(x)  # (B, L, Dproj)
        return x_proj
    
"""
Single conv block: Conv1d(stride=2) -> LayerNorm (over channels) -> GELU -> Optional Dilated/Depthwise conv
Input: (B, L, D) ; We use Conv1d with channels-first.
"""
class ConvDownsampleBlock(nn.Module):
    def __init__(self, dim: int, kernel: int = 5, stride: int = 2, dilation: int = 1, depthwise: bool = False):
        super().__init__()
        padding = (kernel - 1) // 2 * dilation
        if depthwise:
            self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel, stride=stride,
                                       padding=padding, dilation=dilation, groups=dim)
            self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)
        else:
            self.depthwise = None
            self.conv = nn.Conv1d(dim, dim, kernel_size=kernel, stride=stride,
                                  padding=padding, dilation=dilation)
        self.ln = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        x_t = x.transpose(1, 2)  # (B, D, L)
        if self.depthwise is not None:
            y = self.depthwise(x_t)
            y = self.pointwise(y)
        else:
            y = self.conv(x_t)
        y = y.transpose(1, 2)  # (B, L_out, D)
        y = self.ln(y)
        y = self.act(y)
        return y

class ConvDownsampleStack(nn.Module):
    """
    Stack of downsample blocks. Returns compressed features and optionally saved intermediate feature maps.
    """
    def __init__(self, dim: int, n_blocks: int = 3, kernel: int = 5, stride_per_block: int = 2,
                 dilations: Optional[list] = None, depthwise: bool = False, save_intermediate: bool = False):
        super().__init__()
        dilations = dilations or [1] * n_blocks
        self.blocks = nn.ModuleList([
            ConvDownsampleBlock(dim, kernel=kernel, stride=stride_per_block, dilation=dilations[i], depthwise=depthwise)
            for i in range(n_blocks)
        ])
        self.save_intermediate = save_intermediate

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[list]]:
        # x: (B, L, D)
        intermediates = [] if self.save_intermediate else None
        out = x
        for blk in self.blocks:
            out = blk(out)
            if self.save_intermediate:
                intermediates.append(out)
        # out: (B, L', D)
        return out, intermediates