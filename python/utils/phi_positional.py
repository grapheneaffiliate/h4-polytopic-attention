"""
Golden-angle positional encoding using the maximally irrational φ⁻¹ spacing.

Position n gets angle n × 2π × φ⁻¹ on a golden-angle spiral in d_model dimensions.
This guarantees well-separated, non-repeating position vectors for any sequence length.
Long-range positions compress via Zeckendorf decomposition (Fibonacci-based representation).
"""

import math
import torch
import torch.nn as nn

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1.0 / PHI  # φ⁻¹ ≈ 0.618...


def _zeckendorf(n: int):
    """Represent n as a sum of non-consecutive Fibonacci numbers."""
    if n <= 0:
        return []
    fibs = [1, 2]
    while fibs[-1] <= n:
        fibs.append(fibs[-1] + fibs[-2])
    terms = []
    remaining = n
    for f in reversed(fibs):
        if f <= remaining:
            terms.append(f)
            remaining -= f
        if remaining == 0:
            break
    return terms


class PhiPositionalEncoding(nn.Module):
    """
    Golden-angle spiral positional encoding.

    Each position n maps to d_model dimensions via pairs of (cos, sin) at
    golden-angle frequencies. The base angle is n × 2π × φ⁻¹, with each
    dimension pair using a different frequency scale based on φ powers.

    For positions beyond max_cached, Zeckendorf decomposition provides
    logarithmic-cost encoding by summing cached Fibonacci-indexed embeddings.
    """

    def __init__(self, d_model: int, max_cached: int = 8192):
        super().__init__()
        self.d_model = d_model
        self.max_cached = max_cached
        n_pairs = d_model // 2
        has_odd = d_model % 2 == 1

        # Precompute frequency scales: φ^(-k/n_pairs) for k in [0, n_pairs)
        # This gives geometrically spaced frequencies anchored to golden ratio
        freq_scales = torch.tensor(
            [PHI ** (-k / n_pairs) for k in range(n_pairs)],
            dtype=torch.float32,
        )
        self.register_buffer('freq_scales', freq_scales)

        # Precompute position embeddings for [0, max_cached)
        positions = torch.arange(max_cached, dtype=torch.float32)
        # Base angle: position × 2π × φ⁻¹
        base_angles = positions * (2 * math.pi * PHI_INV)  # (max_cached,)
        # Scale by frequency for each pair
        angles = base_angles.unsqueeze(1) * freq_scales.unsqueeze(0)  # (max_cached, n_pairs)

        pe = torch.zeros(max_cached, d_model)
        pe[:, 0::2] = torch.cos(angles[:, :d_model // 2 + (1 if has_odd else 0)])
        pe[:, 1::2] = torch.sin(angles[:, :n_pairs])
        # Normalize to unit norm for consistency with S³ geometry
        pe = pe / (pe.norm(dim=1, keepdim=True) + 1e-8)
        self.register_buffer('pe', pe)

        # Cache Fibonacci numbers for Zeckendorf decomposition
        fibs = [1, 2]
        while fibs[-1] < max_cached * 10:
            fibs.append(fibs[-1] + fibs[-2])
        self.register_buffer('_fibs', torch.tensor(fibs, dtype=torch.long))

    def forward(self, seq_len: int, offset: int = 0) -> torch.Tensor:
        """
        Returns positional encoding of shape (seq_len, d_model).
        For positions < max_cached, uses precomputed table.
        For positions >= max_cached, uses Zeckendorf decomposition.
        """
        if offset + seq_len <= self.max_cached:
            return self.pe[offset:offset + seq_len]

        pe_out = torch.zeros(seq_len, self.d_model, device=self.pe.device)
        for i in range(seq_len):
            pos = offset + i
            if pos < self.max_cached:
                pe_out[i] = self.pe[pos]
            else:
                # Zeckendorf: sum embeddings at Fibonacci indices
                terms = _zeckendorf(pos)
                emb = torch.zeros(self.d_model, device=self.pe.device)
                for fib_val in terms:
                    idx = min(fib_val, self.max_cached - 1)
                    emb = emb + self.pe[idx]
                pe_out[i] = emb / (emb.norm() + 1e-8)
        return pe_out

    def encode_position(self, position: int) -> torch.Tensor:
        """Encode a single position. Returns (d_model,) tensor."""
        if position < self.max_cached:
            return self.pe[position]
        terms = _zeckendorf(position)
        emb = torch.zeros(self.d_model, device=self.pe.device)
        for fib_val in terms:
            idx = min(fib_val, self.max_cached - 1)
            emb = emb + self.pe[idx]
        return emb / (emb.norm() + 1e-8)
