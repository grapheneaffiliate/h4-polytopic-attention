"""
H4 Hybrid Attention Layer — Drop-in replacement for standard transformer attention.

Frozen: 600-cell vertices, H4 simple roots, E8→H4 projection, ChamberTree structure.
Trainable: Q/K/V projections, per-head nudge matrices, chamber bonuses, output projection.

The ChamberTree provides O(log t) candidate filtering. Attention scores are computed
only over top-k candidates per query, giving O(k) attention instead of O(t) softmax.
Gradients flow through the trainable projections and scores; the tree is just a fast filter.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h4_polytopic_attention import generate_600_cell_vertices, build_coxeter_chambers
from utils.chamber_index import ChamberIndex, compute_chamber_ids, chamber_utilization
from bitlinear import BitLinear

PHI = (1 + math.sqrt(5)) / 2


def _build_e8_h4_projection() -> np.ndarray:
    """Build the 4x8 E8→H4 projection matrix from golden ratio eigenvalues."""
    c = math.cos(math.pi / 5)    # φ/2
    s = math.sin(math.pi / 5)
    c2 = math.cos(2 * math.pi / 5)  # 1/(2φ)
    s2 = math.sin(2 * math.pi / 5)
    return np.array([
        [c, s, c2, s2, 0, 0, 0, 0],
        [-s, c, -s2, c2, 0, 0, 0, 0],
        [0, 0, 0, 0, c, s, c2, s2],
        [0, 0, 0, 0, -s, c, -s2, c2],
    ], dtype=np.float64)


class H4AttentionLayer(nn.Module):
    """
    H4 geometric attention: frozen polytopic structure + trainable adapters.

    FROZEN (registered as buffers, no gradients):
        - 600-cell vertices: (120, 4)
        - H4 simple roots: (4, 4)
        - E8→H4 projection: (4, 8)

    TRAINABLE:
        - W_q_proj: Linear(d_model, 4 * n_heads) — project to H4 query space
        - W_k_proj: Linear(d_model, 4 * n_heads) — project to H4 key space
        - W_v_proj: Linear(d_model, d_value * n_heads) — value projection
        - W_nudge: Parameter(n_heads, 4, 4) — per-head query rotation
        - W_out: Linear(d_value * n_heads, d_model) — output projection
        - chamber_bonus: Parameter(n_heads, 16) — per-chamber attention bonus

    Forward pass:
        1. Project input → Q, K, V per head
        2. Normalize Q, K to unit sphere (S³)
        3. Apply W_nudge per head (small rotation of query direction)
        4. Use ChamberTree for O(log t) top-k candidate lookup
        5. Compute attention only over candidates (not full sequence)
        6. Apply chamber_bonus based on query chamber
        7. Weighted sum of values → concatenate heads → W_out
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_value: int = 16,
        top_k: int = 32,
        dropout: float = 0.0,
        use_bitlinear: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = 4  # H4 lives in R^4
        self.d_value = d_value
        self.top_k = top_k
        self.use_bitlinear = use_bitlinear

        # Select layer type: BitLinear (ternary) or nn.Linear (float)
        Linear = BitLinear if use_bitlinear else nn.Linear

        # --- Frozen geometric constants (always float32) ---
        vertices = generate_600_cell_vertices()
        chambers = build_coxeter_chambers(vertices)
        simple_roots = chambers['simple_roots']

        self.register_buffer('vertices', torch.tensor(vertices, dtype=torch.float32))
        self.register_buffer('simple_roots', torch.tensor(simple_roots, dtype=torch.float32))
        self.register_buffer('e8_h4_proj', torch.tensor(_build_e8_h4_projection(), dtype=torch.float32))

        # Store numpy roots for ChamberIndex
        self._simple_roots_np = simple_roots.copy()

        # --- Trainable projections (ternary when use_bitlinear=True) ---
        self.W_q_proj = Linear(d_model, self.d_head * n_heads, bias=False)
        self.W_k_proj = Linear(d_model, self.d_head * n_heads, bias=False)
        self.W_v_proj = Linear(d_model, d_value * n_heads, bias=False)
        self.W_out = Linear(d_value * n_heads, d_model, bias=False)

        # Per-head query nudge: small rotation in H4 space
        # Initialize near identity (small random perturbation)
        self.W_nudge = nn.Parameter(
            torch.eye(4).unsqueeze(0).repeat(n_heads, 1, 1)
            + 0.01 * torch.randn(n_heads, 4, 4)
        )

        # Per-head chamber bonus: stays float — too small (n_heads × 16)
        # to quantize and needs continuous gradients for soft assignment
        self.chamber_bonus = nn.Parameter(torch.zeros(n_heads, 16))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = 1.0 / math.sqrt(self.d_head)

        # Diagnostics (populated during forward, not saved)
        self._last_scan_ratio = 0.0
        self._last_chamber_stats = None

    def forward(
        self,
        x: torch.Tensor,
        use_tree: bool = True,
        return_diagnostics: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            use_tree: if True, use ChamberTree for O(log t) lookup.
                      if False, compute full attention (for comparison/debugging).
            return_diagnostics: if True, return (output, diagnostics_dict)

        Returns:
            output: (batch, seq_len, d_model)
        """
        B, T, D = x.shape

        # Project to Q, K, V
        Q = self.W_q_proj(x).view(B, T, self.n_heads, self.d_head)  # (B, T, H, 4)
        K = self.W_k_proj(x).view(B, T, self.n_heads, self.d_head)
        V = self.W_v_proj(x).view(B, T, self.n_heads, self.d_value)  # (B, T, H, d_v)

        # Normalize Q, K to unit sphere S³
        Q = F.normalize(Q, dim=-1)
        K = F.normalize(K, dim=-1)

        # Apply W_nudge to queries: (B, T, H, 4) @ (H, 4, 4) → (B, T, H, 4)
        Q_nudged = torch.einsum('bthd,hde->bthe', Q, self.W_nudge)
        Q_nudged = F.normalize(Q_nudged, dim=-1)

        if use_tree and T > self.top_k * 2:
            output = self._forward_tree(Q_nudged, K, V, B, T)
        else:
            output = self._forward_full(Q_nudged, K, V, B, T)

        # Concatenate heads and project out
        output = output.reshape(B, T, self.n_heads * self.d_value)
        output = self.W_out(output)
        output = self.dropout(output)

        if return_diagnostics:
            diag = self._compute_diagnostics(Q_nudged, K)
            return output, diag
        return output

    def _forward_full(self, Q, K, V, B, T):
        """Standard full attention (for short sequences or comparison)."""
        # Q, K: (B, T, H, 4), V: (B, T, H, d_v)
        # Compute attention scores: (B, H, T, T)
        scores = torch.einsum('bqhd,bkhd->bhqk', Q, K) * self.scale

        # Add chamber bonus based on KEY chambers (not query, since softmax is shift-invariant)
        # Soft chamber membership of keys via dot products with roots
        k_dots = torch.einsum('bthd,rd->bthr', K, self.simple_roots)  # (B, T, H, 4)
        soft_signs = torch.sigmoid(k_dots * 3.0)  # soft sign for gradient flow
        # Expand to 16 chamber weights
        chamber_weights = torch.ones(B, T, self.n_heads, 16, device=Q.device)
        for bit in range(4):
            s = soft_signs[..., bit].unsqueeze(-1)  # (B, T, H, 1)
            mask = ((torch.arange(16, device=Q.device) >> bit) & 1).float()  # (16,)
            chamber_weights = chamber_weights * (s * mask + (1 - s) * (1 - mask))
        # Weighted sum of chamber bonuses per key: (B, T, H, 16) @ (H, 16) → (B, T, H)
        k_bonus = (chamber_weights * self.chamber_bonus.unsqueeze(0).unsqueeze(0)).sum(-1)
        # Add to scores: each key position gets its chamber-dependent bonus
        # scores: (B, H, T_q, T_k), k_bonus: (B, T_k, H) → (B, H, 1, T_k)
        scores = scores + k_bonus.permute(0, 2, 1).unsqueeze(2)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=Q.device, dtype=torch.bool), diagonal=1
        )
        scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values: (B, H, T, T) @ (B, H, T, d_v) → (B, H, T, d_v)
        V_t = V.permute(0, 2, 1, 3)  # (B, H, T, d_v)
        out = torch.matmul(attn, V_t)  # (B, H, T, d_v)
        return out.permute(0, 2, 1, 3)  # (B, T, H, d_v)

    def _forward_tree(self, Q, K, V, B, T):
        """
        Tree-accelerated attention using ChamberTree for candidate filtering.
        O(k) attention per query instead of O(T).
        """
        device = Q.device
        k = min(self.top_k, T)
        output = torch.zeros(B, T, self.n_heads, self.d_value, device=device)

        total_scanned = 0
        total_possible = 0

        for b in range(B):
            # Build ChamberIndex for this sequence
            chamber_idx = ChamberIndex(self.n_heads, self._simple_roots_np)

            # Insert all keys
            # K[b]: (T, H, 4) → need (T, H, 4) for bulk_insert which expects (seq_len, n_heads, 4)
            chamber_idx.bulk_insert(K[b])

            # Query each position (causal: only attend to positions <= current)
            for t in range(T):
                q = Q[b, t]  # (H, 4)
                candidates = chamber_idx.query_topk(
                    q.unsqueeze(0),  # (1, H, 4)
                    k=k,
                    causal_mask_pos=t,
                )
                # candidates[0] is for this query, shape [H][<=k]

                for h in range(self.n_heads):
                    cand_indices = candidates[0][h]
                    if len(cand_indices) == 0:
                        continue

                    n_cand = len(cand_indices)
                    total_scanned += n_cand
                    total_possible += t + 1

                    idx_tensor = torch.tensor(cand_indices, dtype=torch.long, device=device)

                    # Gather candidate K and V
                    k_cand = K[b, idx_tensor, h]  # (n_cand, 4)
                    v_cand = V[b, idx_tensor, h]  # (n_cand, d_v)

                    # Attention score over candidates only
                    q_h = Q[b, t, h]  # (4,)
                    scores = (q_h @ k_cand.T) * self.scale  # (n_cand,)

                    # Add chamber bonus for this query's chamber
                    q_chamber = compute_chamber_ids(
                        q_h.unsqueeze(0), self.simple_roots
                    )[0].item()
                    scores = scores + self.chamber_bonus[h, q_chamber]

                    attn = F.softmax(scores, dim=0)  # (n_cand,)
                    output[b, t, h] = attn @ v_cand  # (d_v,)

        # Track scan ratio for diagnostics
        self._last_scan_ratio = (
            total_scanned / max(total_possible, 1)
        )

        return output

    def _compute_diagnostics(self, Q, K):
        """Compute geometric diagnostic metrics."""
        diag = {}

        # Chamber utilization of keys
        k_chambers = compute_chamber_ids(K, self.simple_roots)
        stats = chamber_utilization(k_chambers)
        diag['chamber_entropy'] = stats['entropy']
        diag['chamber_max_ratio'] = stats['max_ratio']
        self._last_chamber_stats = stats

        # W_nudge analysis per head
        nudge_ranks = []
        geo_alignments = []
        for h in range(self.n_heads):
            # SVD of nudge deviation from identity
            W = self.W_nudge[h].detach()
            delta = W - torch.eye(4, device=W.device)
            s = torch.linalg.svdvals(delta)
            # Effective rank: ratio of largest to second singular value
            if s[1] > 1e-8:
                nudge_ranks.append((s[0] / s[1]).item())
            else:
                nudge_ranks.append(float('inf'))  # rank 1

            # Alignment with 600-cell vertices
            U, S_vals, Vh = torch.linalg.svd(delta)
            dominant_dir = Vh[0]  # dominant direction
            dots = (self.vertices @ dominant_dir).abs()
            geo_alignments.append(dots.max().item())

        diag['nudge_rank'] = nudge_ranks
        diag['geo_alignment'] = geo_alignments
        diag['scan_ratio'] = self._last_scan_ratio

        return diag


class H4TransformerBlock(nn.Module):
    """Standard transformer block with H4 attention instead of softmax."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_value: int = 16,
        d_ffn: int = None,
        top_k: int = 32,
        dropout: float = 0.0,
        use_bitlinear: bool = False,
    ):
        super().__init__()
        if d_ffn is None:
            d_ffn = d_model * 4

        Linear = BitLinear if use_bitlinear else nn.Linear

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = H4AttentionLayer(
            d_model=d_model,
            n_heads=n_heads,
            d_value=d_value,
            top_k=top_k,
            dropout=dropout,
            use_bitlinear=use_bitlinear,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            Linear(d_model, d_ffn, bias=False),
            nn.GELU(),
            Linear(d_ffn, d_model, bias=False),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(
        self,
        x: torch.Tensor,
        use_tree: bool = True,
        return_diagnostics: bool = False,
    ) -> torch.Tensor:
        # Pre-norm attention with residual
        if return_diagnostics:
            attn_out, diag = self.attn(self.ln1(x), use_tree=use_tree, return_diagnostics=True)
            x = x + attn_out
            x = x + self.ffn(self.ln2(x))
            return x, diag
        else:
            x = x + self.attn(self.ln1(x), use_tree=use_tree)
            x = x + self.ffn(self.ln2(x))
            return x
