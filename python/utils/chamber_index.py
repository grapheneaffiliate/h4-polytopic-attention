"""
PyTorch-compatible chamber lookup for H4 ChamberTree.

Provides a bridge between PyTorch tensors (gradient-tracked) and the
numpy-based H4ChamberTree (discrete, non-differentiable). The key trick:

  - ChamberTree does fast O(log t) filtering to find top-k candidate keys
  - We return candidate indices back to PyTorch
  - Attention scores are computed only over candidates (differentiable)
  - Gradients flow through Q/K projections and scores, not through the tree

This gives O(k) attention per query where k << t.

If the compiled Rust backend (h4_rust) is available, RustChamberIndex provides
a much faster implementation. Falls back to pure-Python ChamberIndex otherwise.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
import sys
import os

# Rust backend detection — optional, graceful fallback to Python
try:
    import h4_rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from h4_polytopic_attention import H4ChamberTree, build_coxeter_chambers, generate_600_cell_vertices


class ChamberIndex:
    """
    Manages a set of H4ChamberTrees (one per head) and provides
    batch top-k candidate lookup compatible with PyTorch autograd.
    """

    def __init__(self, n_heads: int, simple_roots: np.ndarray):
        self.n_heads = n_heads
        self.simple_roots = simple_roots
        self.trees = [H4ChamberTree(simple_roots) for _ in range(n_heads)]
        self._keys_by_head = [[] for _ in range(n_heads)]  # track insertion order

    def reset(self):
        """Clear all trees and rebuild."""
        self.trees = [H4ChamberTree(self.simple_roots) for _ in range(self.n_heads)]
        self._keys_by_head = [[] for _ in range(self.n_heads)]

    def insert_keys(self, keys: torch.Tensor):
        """
        Insert keys for all heads at current timestep.

        Args:
            keys: (n_heads, 4) tensor of key vectors to insert
        """
        keys_np = keys.detach().cpu().numpy()
        t = len(self._keys_by_head[0])  # current position index
        for h in range(self.n_heads):
            key = keys_np[h]
            # Use position index as both value and timestamp
            self.trees[h].insert(key, np.array([t], dtype=np.float64), t)
            self._keys_by_head[h].append(key.copy())

    def bulk_insert(self, keys: torch.Tensor):
        """
        Insert a full sequence of keys for all heads.

        Args:
            keys: (seq_len, n_heads, 4) tensor of key vectors
        """
        seq_len = keys.shape[0]
        keys_np = keys.detach().cpu().numpy()
        for t in range(seq_len):
            for h in range(self.n_heads):
                key = keys_np[t, h]
                self.trees[h].insert(key, np.array([t], dtype=np.float64), t)
                self._keys_by_head[h].append(key.copy())

    def query_topk(
        self,
        queries: torch.Tensor,
        k: int,
        causal_mask_pos: Optional[int] = None,
    ) -> List[List[List[int]]]:
        """
        For each query, find top-k candidate key indices using ChamberTree.

        Args:
            queries: (n_queries, n_heads, 4) tensor of query vectors
            k: number of candidates per query per head
            causal_mask_pos: if set, only return candidates with index <= this value

        Returns:
            List of shape [n_queries][n_heads][<=k] containing key indices.
            These indices can be used to gather from the full key/value tensors.
        """
        n_queries = queries.shape[0]
        queries_np = queries.detach().cpu().numpy()
        results = []

        for q_idx in range(n_queries):
            head_results = []
            for h in range(self.n_heads):
                query = queries_np[q_idx, h]
                # Query tree for top candidates
                # Request more than k since some may be filtered by causal mask
                tree_results = self.trees[h].query_max_dot(query, k=k * 2)

                indices = []
                for score, value, timestamp in tree_results:
                    t_idx = int(value[0]) if len(value) > 0 else timestamp
                    if causal_mask_pos is not None and t_idx > causal_mask_pos:
                        continue
                    indices.append(t_idx)
                    if len(indices) >= k:
                        break

                # If tree didn't return enough, fall back to scanning
                if len(indices) < k and len(self._keys_by_head[h]) > 0:
                    max_pos = causal_mask_pos if causal_mask_pos is not None else len(self._keys_by_head[h]) - 1
                    all_keys = np.array(self._keys_by_head[h][:max_pos + 1])
                    if len(all_keys) > 0:
                        dots = all_keys @ query
                        sorted_idx = np.argsort(-dots)
                        existing = set(indices)
                        for idx in sorted_idx:
                            if idx not in existing:
                                indices.append(int(idx))
                                existing.add(int(idx))
                            if len(indices) >= k:
                                break

                head_results.append(indices)
            results.append(head_results)

        return results


def compute_chamber_ids(keys: torch.Tensor, simple_roots: torch.Tensor) -> torch.Tensor:
    """
    Compute chamber IDs for a batch of keys (differentiable w.r.t. nothing,
    but useful for logging chamber utilization).

    Args:
        keys: (..., 4) tensor of key vectors
        simple_roots: (4, 4) tensor of H4 simple roots

    Returns:
        (...,) tensor of integer chamber IDs (0-15 for 4-bit sign pattern)
    """
    # Dot products with all 4 roots: (..., 4)
    dots = keys @ simple_roots.T
    # Sign pattern → 4-bit chamber ID
    signs = (dots >= 0).long()
    ids = signs[..., 0] * 8 + signs[..., 1] * 4 + signs[..., 2] * 2 + signs[..., 3]
    return ids


def chamber_utilization(chamber_ids: torch.Tensor, n_chambers: int = 16) -> dict:
    """
    Compute chamber utilization statistics.

    Returns:
        Dict with 'counts' (per-chamber), 'entropy' (Shannon entropy),
        and 'max_ratio' (max/mean ratio, 1.0 = perfectly uniform).
    """
    counts = torch.zeros(n_chambers, dtype=torch.long, device=chamber_ids.device)
    flat = chamber_ids.flatten()
    for i in range(n_chambers):
        counts[i] = (flat == i).sum()

    total = counts.sum().float()
    if total == 0:
        return {'counts': counts, 'entropy': 0.0, 'max_ratio': 0.0}

    probs = counts.float() / total
    # Shannon entropy (nats)
    log_probs = torch.where(probs > 0, torch.log(probs), torch.zeros_like(probs))
    entropy = -(probs * log_probs).sum().item()

    mean_count = total / n_chambers
    max_ratio = (counts.max().float() / mean_count).item() if mean_count > 0 else 0.0

    return {
        'counts': counts,
        'entropy': entropy,
        'max_ratio': max_ratio,
    }


class RustChamberIndex:
    """
    Rust-accelerated chamber index using h4_rust compiled backend.
    API-compatible with ChamberIndex for drop-in replacement.

    All heavy computation (dot products, sorting, chamber indexing) runs
    in compiled Rust via PyO3/numpy, typically 10-100x faster than Python.
    """

    def __init__(self, n_heads: int, simple_roots: np.ndarray):
        if not RUST_AVAILABLE:
            raise ImportError("h4_rust is not available. Install with: cd rust && maturin develop --release")
        self.n_heads = n_heads
        self.simple_roots = simple_roots  # (4, 4) numpy array
        self._keys_by_head = [[] for _ in range(n_heads)]  # list of (4,) arrays per head

    def reset(self):
        """Clear all stored keys."""
        self._keys_by_head = [[] for _ in range(self.n_heads)]

    def insert_keys(self, keys: torch.Tensor):
        """
        Insert keys for all heads at current timestep.

        Args:
            keys: (n_heads, 4) tensor of key vectors to insert
        """
        keys_np = keys.detach().cpu().numpy()
        for h in range(self.n_heads):
            self._keys_by_head[h].append(keys_np[h].copy())

    def bulk_insert(self, keys: torch.Tensor):
        """
        Insert a full sequence of keys for all heads.

        Args:
            keys: (seq_len, n_heads, 4) tensor of key vectors
        """
        keys_np = keys.detach().cpu().numpy()
        seq_len = keys_np.shape[0]
        for t in range(seq_len):
            for h in range(self.n_heads):
                self._keys_by_head[h].append(keys_np[t, h].copy())

    def query_topk(
        self,
        queries: torch.Tensor,
        k: int,
        causal_mask_pos: Optional[int] = None,
    ) -> List[List[List[int]]]:
        """
        For each query, find top-k candidate key indices using Rust backend.

        Args:
            queries: (n_queries, n_heads, 4) tensor of query vectors
            k: number of candidates per query per head
            causal_mask_pos: if set, only consider keys with index <= this value

        Returns:
            List of shape [n_queries][n_heads][<=k] containing key indices.
        """
        n_queries = queries.shape[0]
        queries_np = queries.detach().cpu().numpy()
        results = []

        for q_idx in range(n_queries):
            head_results = []
            for h in range(self.n_heads):
                n_keys = len(self._keys_by_head[h])
                if n_keys == 0:
                    head_results.append([])
                    continue

                # Apply causal mask: only use keys up to causal_mask_pos
                max_pos = causal_mask_pos if causal_mask_pos is not None else n_keys - 1
                effective_n = min(n_keys, max_pos + 1)

                if effective_n == 0:
                    head_results.append([])
                    continue

                keys_arr = np.array(self._keys_by_head[h][:effective_n], dtype=np.float64)
                query_arr = queries_np[q_idx, h:h+1].astype(np.float64)

                actual_k = min(k, effective_n)
                indices = h4_rust.query_topk(keys_arr, query_arr, actual_k)
                # indices is (1, actual_k), extract the list and filter -1s
                idx_list = [int(i) for i in indices[0] if i >= 0]
                head_results.append(idx_list)

            results.append(head_results)

        return results


def get_chamber_index(n_heads: int, simple_roots: np.ndarray, prefer_rust: bool = True):
    """
    Factory function: returns RustChamberIndex if available, else ChamberIndex.

    Args:
        n_heads: number of attention heads
        simple_roots: (4, 4) numpy array of H4 simple roots
        prefer_rust: if True (default), use Rust backend when available

    Returns:
        ChamberIndex or RustChamberIndex instance
    """
    if prefer_rust and RUST_AVAILABLE:
        return RustChamberIndex(n_heads, simple_roots)
    return ChamberIndex(n_heads, simple_roots)
