import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from Embedding_CNN import EmbeddingProjection, ConvDownsampleStack
from AttentionPooling import AttentionPooling
from Selector import TopKSelector
from Composer import Composer
from ReconstructionHead import ReconstructionHead

class HybridCompressor(nn.Module):
    """
    API:
      - forward(inputs, tokens_or_embeddings, keep_recent=None, topk=None)
    Returns:
      dict with:
        - transformer_input: (B, L_t, Dmodel)
        - passthrough_idx: indices of passthrough tokens (B, K)
        - pooled_count: M
        - recon_targets (optional)
    """
    def __init__(self,
                 D: int = 1024,
                 Dproj: Optional[int] = None,
                 Dmodel: Optional[int] = None,
                 conv_blocks: int = 3,
                 pool_M: int = 4096,
                 pool_heads: int = 8,
                 topk: int = 4000,
                 depthwise: bool = False,
                 interleave: bool = False):
        super().__init__()
        Dproj = Dproj or D
        Dmodel = Dmodel or D
        self.embed_proj = EmbeddingProjection(vocab_size=None, D=D, Dproj=Dproj, use_embedding_table=False)
        self.conv_stack = ConvDownsampleStack(Dproj, n_blocks=conv_blocks, depthwise=depthwise, save_intermediate=False)
        self.attn_pool = AttentionPooling(Dproj, M=pool_M, n_heads=pool_heads)
        self.selector = TopKSelector(Dproj)
        self.merger = Composer(Dproj, Dmodel, interleave=interleave)
        # optional
        self.recon = ReconstructionHead(Dproj, Dmodel)  
        self.pool_M = pool_M
        self.topk_param = topk

    def forward(self, embeddings: torch.Tensor, topk: Optional[int] = None,
                keep_recent: Optional[int] = None) -> Dict[str, Any]:
        """
        embeddings: (B, L, D)
        topk: number of passthrough tokens to keep from original features (default self.topk_param)
        keep_recent: number of last tokens to always passthrough (for autoreg inference).
        """
        B, L, D = embeddings.shape
        x = self.embed_proj.proj(embeddings)  # (B, L, Dproj)
        if keep_recent is None:
            keep_recent = 0
        keep_recent = max(0, min(keep_recent, L))
        # compute conv downsample
        conv_feats, intermediates = self.conv_stack(x)  # (B, L', Dproj)
        # pooling
        pooled = self.attn_pool(conv_feats)  # (B, M, Dproj)
        # selection: run scorer on *original* embeddings (or use conv_feats upsampled)
        k = topk or self.topk_param
        passthrough = embeddings.new_empty((B, 0, D))
        idx_full = torch.empty((B, 0), dtype = torch.long, device = embeddings.device)
            # if keep_recent equals or exceeds L, everything is forced passthrough -- skip selector
        if keep_recent >= L:
            forced = embeddings  # (B, L, D)
            forced_idx = torch.arange(0, L, device = embeddings.device).unsqueeze(0).expand(B, -1)
            passthrough = forced
            idx_full = forced_idx
        else:
            if keep_recent > 0:
                forced = embeddings[:, -keep_recent:, :]  # (B, W, D)
                forced_idx = torch.arange(L - keep_recent, L, device=embeddings.device).unsqueeze(0).expand(B, -1)  # (B, W)
                # for the rest, compute scores
                rest_feats = x[:, :L - keep_recent, :]  # (B, L-W, Dproj)
                k_remaining = max(0, k - keep_recent)
                if k_remaining > 0 and rest_feats.shape[1] > 0:
                    sel, scores, idx = self.selector(rest_feats, k_remaining)
                    # build passthrough
                    if idx.numel() > 0:
                        batch_idx = torch.arange(B, device = embeddings.device).unsqueeze(-1)
                        sel_gather = rest_feats[batch_idx, idx, :]  # (B, k_remaining, D)
                        global_idx = idx
                        orig_selected = embeddings[batch_idx, global_idx, :]  # (B, k_remaining, D)
                        passthrough = torch.cat([orig_selected, forced], dim=1)  # (B, k, D)
                        # shift idx to be relative to original embeddings (rest starts at 0)
                        idx_full = torch.cat([global_idx, forced_idx], dim = 1)  # already relative to rest_feats start=0
                    else:
                        passthrough = forced
                        idx_full = forced_idx
                else:
                    passthrough = forced
                    idx_full = forced_idx
            else:
                if L == 0 or k == 0:
                    passthrough = embeddings.new_empty((B, 0, D))
                    idx_full = torch.empty((B, 0), dtype = torch.long, device = embeddings.device)
                else:
                    sel, scores, idx = self.selector(x, k)
                    if idx.dim() == 1:
                        idx = idx.unsqueeze(0).expand(B, -1)
                    batch_idx = torch.arange(B, device = embeddings.device).unsqueeze(-1)
                    if idx.numel() > 0:
                        orig_selected = embeddings[batch_idx, idx, :]
                        passthrough = orig_selected
                        idx_full = idx
                    else:
                        passthrough = embeddings.new_empty((B, 0, D))
                        idx_full = torch.empty((B, 0), dtype = torch.long, device = embeddings.device)
                        
        if passthrough.dim() == 2:
            passthrough = passthrough.unsqueeze(0)
        if passthrough.shape[1] != idx_full.shape[1]:
            min_k = min(passthrough.shape[1], idx_full.shape[1])
            idx_full = idx_full[:, :min_k]
            passthrough = passthrough[:, :min_k, :]
        # gating: compute gate on passthrough features (optional) - for simplicity we skip mixing here
        # merge pooled + passthrough
        transformer_input = self.merger(pooled, passthrough)
        out = {
            "transformer_input": transformer_input,  # (B, L_t, Dmodel) where L_t = M + K or K + M
            "passthrough_idx": idx_full,
            "pooled_count": self.pool_M,
            "passthrough_count": passthrough.shape[1],
            "conv_feats": conv_feats  # optional for reconstruction
        }
        return out