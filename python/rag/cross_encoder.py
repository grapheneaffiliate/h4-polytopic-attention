"""
H4 Cross-Encoder Reranker — score (question, passage) with full cross-attention.

The bi-encoder (ranking_model.py) encodes question and passage separately.
Fast but can't compare them directly — gets R@5=100% but R@1 plateaus at ~40%.

The cross-encoder feeds question + passage as ONE sequence through H4 attention.
The attention heads directly attend from question tokens to passage tokens.
Slower (one forward pass per candidate) but much more precise.

Production pipeline:
  1. Bi-encoder retrieves top-k candidates (fast, 20ms for all docs)
  2. Cross-encoder reranks k candidates (precise, ~10ms per candidate)
  3. Return the top-ranked candidate

Uses the PPL 10.0 TinyStories checkpoint as backbone — the model already
knows English, it just needs to learn "does this passage answer this question?"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from h4_language_model import H4LanguageModel
from bitlinear import BitLinear


class H4CrossEncoder(nn.Module):
    """
    Cross-encoder reranker using H4 attention.

    Input: [question tokens] [SEP] [passage tokens]
    Output: scalar relevance score

    The H4 attention heads attend across the full concatenated sequence,
    so question tokens can directly attend to passage tokens via
    ChamberTree geometric routing.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        use_bitlinear: bool = True,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.d_model = d_model

        # Use the same architecture as the language model
        self.lm = H4LanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_value=d_model // n_heads,
            d_ffn=d_model * 4,
            max_seq_len=max_seq_len,
            dropout=0.0,
            use_bitlinear=use_bitlinear,
        )

        # Classification head: pool the sequence, project to scalar
        Linear = BitLinear if use_bitlinear else nn.Linear
        self.score_head = nn.Sequential(
            Linear(d_model, d_model // 4, bias=False),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

    def load_lm_backbone(self, checkpoint_path: str):
        """
        Load pre-trained language model weights as backbone.
        The LM head is discarded; we keep the transformer blocks.
        """
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        lm_state = ckpt['model_state']
        model_state = self.lm.state_dict()

        loaded = 0
        skipped = 0
        for key in lm_state:
            if key in model_state and lm_state[key].shape == model_state[key].shape:
                model_state[key] = lm_state[key]
                loaded += 1
            else:
                skipped += 1

        self.lm.load_state_dict(model_state)
        print(f"Loaded LM backbone: {loaded} tensors, {skipped} skipped")
        return ckpt.get('config', {})

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Score a batch of (question + passage) sequences.

        Args:
            input_ids: (B, T) — tokenized [question SEP passage]
        Returns:
            (B,) relevance scores
        """
        # Get transformer hidden states (bypass LM head)
        B, T = input_ids.shape
        tok_emb = self.lm.token_emb(input_ids) * self.lm.emb_scale
        pos_emb = self.lm.pos_enc(T).unsqueeze(0).to(tok_emb.device)
        x = self.lm.emb_dropout(tok_emb + pos_emb)

        for block in self.lm.blocks:
            x = block(x, use_tree=False)

        x = self.lm.ln_f(x)

        # Mean pool over non-padding tokens
        pad_mask = (input_ids != 0).float().unsqueeze(-1)  # (B, T, 1)
        pooled = (x * pad_mask).sum(dim=1) / pad_mask.sum(dim=1).clamp(min=1)  # (B, d_model)

        # Score
        score = self.score_head(pooled).squeeze(-1)  # (B,)
        return score

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
