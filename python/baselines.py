"""
Baseline attention mechanisms for comparison with H4 Polytopic Attention.

Implements standard softmax attention and linear attention (Katharopoulos et al. 2020)
with the SAME model wrapper (embeddings, FFN, LM head) so the only variable is attention.

Usage:
    model = BaselineLanguageModel(vocab_size=128, d_model=128, n_heads=8,
                                   n_layers=4, d_value=16, d_ffn=512,
                                   attention_type='softmax')  # or 'linear'
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.phi_positional import PhiPositionalEncoding
from bitlinear import BitLinear


# ---------------------------------------------------------------------------
# Softmax Attention (standard transformer)
# ---------------------------------------------------------------------------

class SoftmaxAttention(nn.Module):
    """Standard multi-head scaled dot-product attention with causal mask."""

    def __init__(self, d_model, n_heads, d_value, dropout=0.0, use_bitlinear=False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_value = d_value
        self.scale = 1.0 / math.sqrt(self.d_head)

        Linear = BitLinear if use_bitlinear else nn.Linear

        self.W_q = Linear(d_model, self.d_head * n_heads, bias=False)
        self.W_k = Linear(d_model, self.d_head * n_heads, bias=False)
        self.W_v = Linear(d_model, d_value * n_heads, bias=False)
        self.W_out = Linear(d_value * n_heads, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, **kwargs):
        B, T, D = x.shape
        H = self.n_heads

        Q = self.W_q(x).view(B, T, H, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        K = self.W_k(x).view(B, T, H, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, self.d_value).transpose(1, 2)  # (B, H, T, d_value)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, H, T, d_value)
        out = out.transpose(1, 2).contiguous().view(B, T, H * self.d_value)
        return self.W_out(out)


# ---------------------------------------------------------------------------
# Linear Attention (Katharopoulos et al. 2020)
# ---------------------------------------------------------------------------

def elu_feature_map(x):
    """ELU+1 feature map for linear attention: phi(x) = elu(x) + 1."""
    return F.elu(x) + 1.0


class LinearAttention(nn.Module):
    """
    Linear attention: O(T) causal attention via kernel trick.

    Instead of softmax(QK^T)V, computes phi(Q) @ (phi(K)^T @ V)
    where phi is the ELU+1 feature map.

    For causal attention, uses cumulative sum formulation:
        S_t = sum_{i<=t} phi(K_i)^T V_i   (running state)
        z_t = sum_{i<=t} phi(K_i)          (running normalizer)
        output_t = (phi(Q_t) @ S_t) / (phi(Q_t) @ z_t)
    """

    def __init__(self, d_model, n_heads, d_value, dropout=0.0, use_bitlinear=False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_value = d_value

        Linear = BitLinear if use_bitlinear else nn.Linear

        self.W_q = Linear(d_model, self.d_head * n_heads, bias=False)
        self.W_k = Linear(d_model, self.d_head * n_heads, bias=False)
        self.W_v = Linear(d_model, d_value * n_heads, bias=False)
        self.W_out = Linear(d_value * n_heads, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, **kwargs):
        B, T, D = x.shape
        H = self.n_heads

        Q = self.W_q(x).view(B, T, H, self.d_head)  # (B, T, H, d_head)
        K = self.W_k(x).view(B, T, H, self.d_head)
        V = self.W_v(x).view(B, T, H, self.d_value)  # (B, T, H, d_value)

        # Apply ELU+1 feature map
        Q = elu_feature_map(Q)  # (B, T, H, d_head)
        K = elu_feature_map(K)

        # Causal linear attention via cumulative sum
        # S_t = cumsum(phi(K)^T @ V) over time dimension
        # For each timestep: KV = outer(K_t, V_t) -> (B, H, d_head, d_value)
        # Cumulative: S_t = sum_{i<=t} KV_i

        # Reshape for batch computation
        Q = Q.permute(0, 2, 1, 3)  # (B, H, T, d_head)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)  # (B, H, T, d_value)

        # KV: outer product at each timestep
        KV = torch.einsum('bhti,bhtj->bhtij', K, V)  # (B, H, T, d_head, d_value)
        S = torch.cumsum(KV, dim=2)  # (B, H, T, d_head, d_value)

        # Normalizer: cumsum of K
        z = torch.cumsum(K, dim=2)  # (B, H, T, d_head)

        # Output: Q @ S / (Q @ z)
        # numerator: (B, H, T, d_head) @ (B, H, T, d_head, d_value) -> (B, H, T, d_value)
        num = torch.einsum('bhti,bhtij->bhtj', Q, S)
        # denominator: (B, H, T, d_head) . (B, H, T, d_head) -> (B, H, T)
        den = torch.einsum('bhti,bhti->bht', Q, z).unsqueeze(-1).clamp(min=1e-6)

        out = num / den  # (B, H, T, d_value)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, H * self.d_value)
        out = self.dropout(out)
        return self.W_out(out)


# ---------------------------------------------------------------------------
# Transformer Block (swappable attention)
# ---------------------------------------------------------------------------

class SoftmaxTransformerBlock(nn.Module):
    """Standard pre-norm transformer block with softmax attention."""

    def __init__(self, d_model, n_heads, d_value, d_ffn=None, dropout=0.0,
                 use_bitlinear=False):
        super().__init__()
        if d_ffn is None:
            d_ffn = d_model * 4
        Linear = BitLinear if use_bitlinear else nn.Linear

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SoftmaxAttention(d_model, n_heads, d_value, dropout, use_bitlinear)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            Linear(d_model, d_ffn, bias=False),
            nn.GELU(),
            Linear(d_ffn, d_model, bias=False),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x, **kwargs):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class LinearTransformerBlock(nn.Module):
    """Pre-norm transformer block with linear attention (Katharopoulos et al. 2020)."""

    def __init__(self, d_model, n_heads, d_value, d_ffn=None, dropout=0.0,
                 use_bitlinear=False):
        super().__init__()
        if d_ffn is None:
            d_ffn = d_model * 4
        Linear = BitLinear if use_bitlinear else nn.Linear

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = LinearAttention(d_model, n_heads, d_value, dropout, use_bitlinear)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            Linear(d_model, d_ffn, bias=False),
            nn.GELU(),
            Linear(d_ffn, d_model, bias=False),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x, **kwargs):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Baseline Language Model
# ---------------------------------------------------------------------------

class BaselineLanguageModel(nn.Module):
    """
    Language model with swappable attention mechanism.

    Same architecture as H4LanguageModel (same embeddings, FFN, LM head)
    but with standard softmax or linear attention instead of H4 geometric attention.
    This ensures the only variable in comparisons is the attention mechanism.

    Args:
        vocab_size: vocabulary size
        d_model: model dimension
        n_heads: number of attention heads
        n_layers: number of transformer blocks
        d_value: value dimension per head
        d_ffn: FFN hidden dimension (default: 4 * d_model)
        max_seq_len: max sequence length for positional encoding
        dropout: dropout rate
        attention_type: 'softmax' or 'linear'
        use_bitlinear: if True, use ternary weights
    """

    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_heads=8,
        n_layers=4,
        d_value=16,
        d_ffn=None,
        max_seq_len=512,
        dropout=0.0,
        attention_type='softmax',
        use_bitlinear=False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.attention_type = attention_type

        if d_ffn is None:
            d_ffn = d_model * 4

        # Token embedding (always float)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.emb_scale = math.sqrt(d_model)

        # Same golden-angle positional encoding as H4LanguageModel
        self.pos_enc = PhiPositionalEncoding(d_model, max_cached=max_seq_len)

        self.emb_dropout = nn.Dropout(dropout)

        # Transformer blocks with selected attention type
        if attention_type == 'softmax':
            BlockClass = SoftmaxTransformerBlock
        elif attention_type == 'linear':
            BlockClass = LinearTransformerBlock
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")

        self.blocks = nn.ModuleList([
            BlockClass(
                d_model=d_model,
                n_heads=n_heads,
                d_value=d_value,
                d_ffn=d_ffn,
                dropout=dropout,
                use_bitlinear=use_bitlinear,
            )
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)

        # LM head (tied with token embedding)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following GPT-2 conventions."""
        for module in self.modules():
            if isinstance(module, BitLinear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, **kwargs):
        """
        Args:
            input_ids: (batch, seq_len) token indices
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = input_ids.shape

        tok_emb = self.token_emb(input_ids) * self.emb_scale
        pos_emb = self.pos_enc(T).unsqueeze(0).to(tok_emb.device)
        x = self.emb_dropout(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
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
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k_sample=0):
        """Autoregressive generation."""
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature

            if top_k_sample > 0:
                v, _ = torch.topk(logits, min(top_k_sample, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

        return input_ids
