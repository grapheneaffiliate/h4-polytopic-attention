"""
H4 Geometric Ranker: score (question, passage) relevance in H4 space.

Architecture:
    1. Encode question with H4 attention (4D geometric heads)
    2. Encode passage with H4 attention (shared weights)
    3. Relevance = dot product on S³ (same metric as ChamberTree attention)

The scoring uses the SAME geometry as attention routing.
No separate scoring function needed — the architecture is the scorer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from h4_hybrid_attention import H4TransformerBlock
from bitlinear import BitLinear


class H4Ranker(nn.Module):
    """
    Score (question, passage) relevance via H4 geometric similarity.

    Both question and passage are encoded to 4D vectors on S³.
    Relevance = dot product in H4 space. Higher = more relevant.
    Trained with contrastive loss (InfoNCE) using in-batch negatives.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        d_value: int = 16,
        d_ffn: int = None,
        use_bitlinear: bool = True,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_bitlinear = use_bitlinear

        if d_ffn is None:
            d_ffn = d_model * 4

        Linear = BitLinear if use_bitlinear else nn.Linear

        # Token embedding (shared between question and passage)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.emb_scale = math.sqrt(d_model)

        # H4 attention blocks (shared encoder)
        self.blocks = nn.ModuleList([
            H4TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_value=d_value,
                d_ffn=d_ffn,
                dropout=0.0,
                use_bitlinear=use_bitlinear,
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        # Project from d_model to 4D (H4 space) for geometric scoring
        self.to_h4 = Linear(d_model, 4, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, BitLinear)):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of sequences to 4D vectors on S³.

        Args:
            token_ids: (B, T) tokenized text (0-padded)
        Returns:
            (B, 4) unit vectors in H4 space
        """
        # Create padding mask
        pad_mask = (token_ids != 0).float()  # (B, T)

        x = self.embedding(token_ids) * self.emb_scale  # (B, T, d_model)

        for block in self.blocks:
            x = block(x, use_tree=False)

        x = self.ln_f(x)

        # Masked mean pool (ignore padding)
        mask = pad_mask.unsqueeze(-1)  # (B, T, 1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, d_model)

        # Project to H4 and normalize to S³
        h4 = self.to_h4(x)  # (B, 4)
        h4 = F.normalize(h4, dim=-1)

        return h4

    def score(self, q_ids: torch.Tensor, p_ids: torch.Tensor) -> torch.Tensor:
        """
        Score relevance of (question, passage) pairs.

        Args:
            q_ids: (B, T_q) question tokens
            p_ids: (B, T_p) passage tokens
        Returns:
            (B,) scores in [-1, 1]
        """
        q_h4 = self.encode(q_ids)
        p_h4 = self.encode(p_ids)
        return (q_h4 * p_h4).sum(dim=-1)

    def count_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable
