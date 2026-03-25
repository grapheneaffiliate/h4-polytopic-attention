"""
H4 Language Model — Transformer LM with H4 geometric attention.

Architecture:
    - Token embedding + golden-angle positional encoding (PhiPositionalEncoding)
    - N × H4TransformerBlock (H4 attention + FFN)
    - LM head (Linear to vocab_size)

The frozen H4 geometry handles spatial partitioning of attention space.
Trainable adapters (nudge matrices, chamber bonuses, projections) learn
which directions to query and how to weight chambers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h4_hybrid_attention import H4TransformerBlock
from utils.phi_positional import PhiPositionalEncoding
from bitlinear import BitLinear


class H4LanguageModel(nn.Module):
    """
    Full language model with H4 polytopic attention.

    Args:
        vocab_size: vocabulary size
        d_model: model dimension
        n_heads: number of H4 attention heads per layer
        n_layers: number of transformer blocks
        d_value: value dimension per head
        d_ffn: FFN hidden dimension (default: 4 * d_model)
        top_k: max candidates per query in ChamberTree lookup
        max_seq_len: max sequence length for positional encoding cache
        dropout: dropout rate
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 8,
        n_layers: int = 4,
        d_value: int = 16,
        d_ffn: int = None,
        top_k: int = 32,
        max_seq_len: int = 8192,
        dropout: float = 0.1,
        use_bitlinear: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_bitlinear = use_bitlinear

        if d_ffn is None:
            d_ffn = d_model * 4

        # Token embedding (always float — lookup table, not a matmul)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # Scale embedding by sqrt(d_model) as in original transformer
        self.emb_scale = math.sqrt(d_model)

        # Golden-angle positional encoding
        self.pos_enc = PhiPositionalEncoding(d_model, max_cached=max_seq_len)

        # Embedding dropout
        self.emb_dropout = nn.Dropout(dropout)

        # Transformer blocks with H4 attention
        self.blocks = nn.ModuleList([
            H4TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_value=d_value,
                d_ffn=d_ffn,
                top_k=top_k,
                dropout=dropout,
                use_bitlinear=use_bitlinear,
            )
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)

        # LM head (tied with token embedding weights — stays float)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following GPT-2 conventions."""
        for module in self.modules():
            if isinstance(module, BitLinear):
                # BitLinear already has kaiming init; apply GPT-2 scale
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        use_tree: bool = True,
        return_diagnostics: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token indices
            use_tree: if True, use ChamberTree for O(log t) attention
            return_diagnostics: if True, return (logits, list_of_diag_dicts)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = input_ids.shape

        # Token + positional embedding
        tok_emb = self.token_emb(input_ids) * self.emb_scale  # (B, T, D)
        pos_emb = self.pos_enc(T).unsqueeze(0).to(tok_emb.device)  # (1, T, D)
        x = self.emb_dropout(tok_emb + pos_emb)

        # Transformer blocks
        diagnostics = []
        for block in self.blocks:
            if return_diagnostics:
                x, diag = block(x, use_tree=use_tree, return_diagnostics=True)
                diagnostics.append(diag)
            else:
                x = block(x, use_tree=use_tree)

        # Final norm + LM head
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if return_diagnostics:
            return logits, diagnostics
        return logits

    def count_params(self):
        """Count trainable and frozen parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        buffers = sum(b.numel() for b in self.buffers())
        return {
            'trainable': trainable,
            'frozen': frozen,
            'buffers': buffers,
            'total': trainable + frozen,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k_sample: int = 0,
    ) -> torch.Tensor:
        """Autoregressive generation."""
        for _ in range(max_new_tokens):
            # Crop to max sequence length if needed
            logits = self.forward(input_ids, use_tree=False)
            logits = logits[:, -1, :] / temperature

            if top_k_sample > 0:
                v, _ = torch.topk(logits, min(top_k_sample, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

        return input_ids
