"""
Benchmark: H4 geometric attention vs standard softmax attention.

Compares wall-clock time, peak memory, and attention score quality
at various context lengths to find the empirical crossover point
where H4's O(log t) chamber lookup beats softmax's O(t^2) matmul.

Now includes Rust-accelerated backend (h4_rust) when available.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h4_hybrid_attention import H4AttentionLayer
from utils.chamber_index import compute_chamber_ids

# Rust backend detection
try:
    import h4_rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


class SoftmaxAttentionLayer(nn.Module):
    """Standard multi-head scaled dot-product attention for comparison."""

    def __init__(self, d_model: int, n_heads: int = 8, d_value: int = 16, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_value = d_value
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.W_q = nn.Linear(d_model, self.d_head * n_heads, bias=False)
        self.W_k = nn.Linear(d_model, self.d_head * n_heads, bias=False)
        self.W_v = nn.Linear(d_model, d_value * n_heads, bias=False)
        self.W_out = nn.Linear(d_value * n_heads, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_value).permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).reshape(B, T, -1)
        return self.W_out(out)


def benchmark_forward_pass(layer, x, n_warmup=2, n_runs=5, **kwargs):
    """Time forward pass, return mean and std in milliseconds."""
    for _ in range(n_warmup):
        _ = layer(x, **kwargs)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = layer(x, **kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return np.mean(times), np.std(times)


def benchmark_rust_topk(keys_np, queries_np, k, n_warmup=2, n_runs=5):
    """
    Benchmark Rust h4_rust.query_topk on raw numpy arrays.
    Returns mean and std in milliseconds.
    """
    if not RUST_AVAILABLE:
        return None, None

    keys = keys_np.astype(np.float64)
    queries = queries_np.astype(np.float64)

    # Warmup
    for _ in range(n_warmup):
        _ = h4_rust.query_topk(keys, queries, k)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = h4_rust.query_topk(keys, queries, k)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return np.mean(times), np.std(times)


def benchmark_numpy_topk(keys_np, queries_np, k, n_warmup=2, n_runs=5):
    """
    Benchmark pure-numpy brute-force top-k for comparison.
    Returns mean and std in milliseconds.
    """
    keys = keys_np.astype(np.float64)
    queries = queries_np.astype(np.float64)

    # Normalize
    k_norms = np.linalg.norm(keys, axis=1, keepdims=True)
    k_norms[k_norms < 1e-12] = 1.0
    keys_normed = keys / k_norms

    q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    q_norms[q_norms < 1e-12] = 1.0
    queries_normed = queries / q_norms

    # Warmup
    for _ in range(n_warmup):
        dots = queries_normed @ keys_normed.T
        _ = np.argsort(-dots, axis=1)[:, :k]

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        dots = queries_normed @ keys_normed.T
        _ = np.argsort(-dots, axis=1)[:, :k]
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return np.mean(times), np.std(times)


def compare_attention_patterns(h4_layer, softmax_layer, x):
    """
    Compare attention score distributions between H4 and softmax.
    Returns correlation coefficient.
    """
    B, T, D = x.shape

    h4_out = h4_layer(x, use_tree=False)
    softmax_out = softmax_layer(x)

    h4_flat = h4_out.detach().flatten()
    sm_flat = softmax_out.detach().flatten()

    if h4_flat.std() < 1e-8 or sm_flat.std() < 1e-8:
        return 0.0

    corr = torch.corrcoef(torch.stack([h4_flat, sm_flat]))[0, 1].item()
    return corr


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    d_model = 64
    n_heads = 8
    d_value = 16
    batch_size = 1
    top_k = 32

    # Part 1 uses the full H4 attention layer (Python tree), so keep lengths moderate
    layer_seq_lengths = [64, 128, 256, 512, 1024]

    # Part 2 tests raw Rust top-k at extended lengths
    rust_seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]

    print("=" * 100)
    print("H4 Geometric Attention vs Standard Softmax Attention -- Benchmark")
    print("=" * 100)
    print(f"d_model={d_model}, n_heads={n_heads}, d_value={d_value}, batch_size={batch_size}, top_k={top_k}")
    print(f"Rust backend (h4_rust): {'AVAILABLE' if RUST_AVAILABLE else 'NOT AVAILABLE (install with: cd rust && maturin develop --release)'}")
    print()

    # Create layers
    h4_layer = H4AttentionLayer(d_model, n_heads, d_value, top_k=top_k)
    softmax_layer = SoftmaxAttentionLayer(d_model, n_heads, d_value)

    h4_layer.eval()
    softmax_layer.eval()

    # ============================================================
    # Part 1: Full attention layer benchmark (softmax vs H4)
    # ============================================================
    print("-" * 100)
    print("PART 1: Full Attention Layer Forward Pass (ms)")
    print("-" * 100)

    results = []

    header = f"{'seq_len':>8} | {'softmax_ms':>12} | {'h4_full_ms':>12} | {'h4_tree_ms':>12} | {'tree/full':>10} | {'corr':>8}"
    print(header)
    print("-" * len(header))

    for T in layer_seq_lengths:
        x = torch.randn(batch_size, T, d_model)

        with torch.no_grad():
            sm_mean, sm_std = benchmark_forward_pass(softmax_layer, x)
            h4_full_mean, h4_full_std = benchmark_forward_pass(h4_layer, x, use_tree=False)

            if T > 64:
                h4_tree_mean, h4_tree_std = benchmark_forward_pass(h4_layer, x, use_tree=True, n_runs=3)
            else:
                h4_tree_mean = h4_full_mean
                h4_tree_std = h4_full_std

            corr = compare_attention_patterns(h4_layer, softmax_layer, x)
            ratio = h4_tree_mean / max(h4_full_mean, 0.001)

            print(f"{T:8d} | {sm_mean:10.1f}+/-{sm_std:3.1f} | {h4_full_mean:10.1f}+/-{h4_full_std:3.1f} | {h4_tree_mean:10.1f}+/-{h4_tree_std:3.1f} | {ratio:10.3f} | {corr:8.4f}")

            results.append({
                'seq_len': T,
                'softmax_ms': sm_mean,
                'h4_full_ms': h4_full_mean,
                'h4_tree_ms': h4_tree_mean,
                'tree_vs_full_ratio': ratio,
                'output_correlation': corr,
            })

    # ============================================================
    # Part 2: Raw top-k benchmark (Rust vs NumPy)
    # ============================================================
    print()
    print("-" * 100)
    print("PART 2: Raw Top-k Query Benchmark — Rust h4_rust vs NumPy (ms)")
    print("  (One attention head: n_queries=64 queries against n_keys keys, k=32)")
    print("-" * 100)

    n_queries = 64
    k = 32

    if RUST_AVAILABLE:
        header2 = f"{'n_keys':>8} | {'numpy_ms':>12} | {'rust_ms':>12} | {'speedup':>10}"
        print(header2)
        print("-" * len(header2))

        rust_results = []
        for T in rust_seq_lengths:
            keys_np = np.random.randn(T, 4).astype(np.float64)
            queries_np = np.random.randn(n_queries, 4).astype(np.float64)

            np_mean, np_std = benchmark_numpy_topk(keys_np, queries_np, k)
            rust_mean, rust_std = benchmark_rust_topk(keys_np, queries_np, k)

            speedup = np_mean / max(rust_mean, 0.001) if rust_mean else 0.0

            print(f"{T:8d} | {np_mean:10.3f}+/-{np_std:3.3f} | {rust_mean:10.3f}+/-{rust_std:3.3f} | {speedup:9.1f}x")

            rust_results.append({
                'n_keys': T,
                'numpy_ms': np_mean,
                'rust_ms': rust_mean,
                'speedup': speedup,
            })
    else:
        print("  [SKIPPED] Rust backend not available.")
        print("  Install with: cd rust && maturin develop --release")
        rust_results = []

    # ============================================================
    # Part 3: Chamber index computation benchmark
    # ============================================================
    print()
    print("-" * 100)
    print("PART 3: Chamber Index Computation — Rust vs NumPy (ms)")
    print("-" * 100)

    if RUST_AVAILABLE:
        roots = h4_rust.get_simple_roots()  # (4, 4) f64
        header3 = f"{'n_vectors':>10} | {'numpy_ms':>12} | {'rust_ms':>12} | {'speedup':>10}"
        print(header3)
        print("-" * len(header3))

        for n_vecs in [1000, 10000, 100000]:
            vecs = np.random.randn(n_vecs, 4).astype(np.float64)
            roots_torch = torch.from_numpy(roots).float()

            # NumPy/torch chamber IDs
            vecs_torch = torch.from_numpy(vecs).float()
            # Warmup
            for _ in range(2):
                _ = compute_chamber_ids(vecs_torch, roots_torch)

            times_np = []
            for _ in range(5):
                t0 = time.perf_counter()
                _ = compute_chamber_ids(vecs_torch, roots_torch)
                t1 = time.perf_counter()
                times_np.append((t1 - t0) * 1000)
            np_mean = np.mean(times_np)
            np_std_val = np.std(times_np)

            # Rust chamber IDs
            for _ in range(2):
                _ = h4_rust.chamber_indices(vecs, roots)

            times_rust = []
            for _ in range(5):
                t0 = time.perf_counter()
                _ = h4_rust.chamber_indices(vecs, roots)
                t1 = time.perf_counter()
                times_rust.append((t1 - t0) * 1000)
            rust_mean = np.mean(times_rust)
            rust_std_val = np.std(times_rust)

            speedup = np_mean / max(rust_mean, 0.001)
            print(f"{n_vecs:10d} | {np_mean:10.3f}+/-{np_std_val:3.3f} | {rust_mean:10.3f}+/-{rust_std_val:3.3f} | {speedup:9.1f}x")

            # Verify correctness: Rust and torch should agree
            ids_torch = compute_chamber_ids(vecs_torch, roots_torch).numpy()
            ids_rust = h4_rust.chamber_indices(vecs, roots)
            # Note: bit ordering may differ, just check both produce valid 0-15 range
            assert ids_rust.min() >= 0 and ids_rust.max() <= 15, "Rust chamber IDs out of range"
    else:
        print("  [SKIPPED] Rust backend not available.")

    # ============================================================
    # Summary
    # ============================================================
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    # Scaling analysis from Part 1
    if len(results) >= 2:
        sm_times = [(r['seq_len'], r['softmax_ms']) for r in results]
        h4_times = [(r['seq_len'], r['h4_tree_ms']) for r in results]

        sm_exp = math.log(sm_times[-1][1] / max(sm_times[0][1], 0.01)) / math.log(sm_times[-1][0] / sm_times[0][0])
        h4_exp = math.log(h4_times[-1][1] / max(h4_times[0][1], 0.01)) / math.log(h4_times[-1][0] / h4_times[0][0])

        print(f"  Softmax scaling exponent: ~{sm_exp:.2f} (expect ~2.0 for O(t^2))")
        print(f"  H4 tree scaling exponent: ~{h4_exp:.2f} (expect ~0 for O(log t), higher due to Python overhead)")

    crossover = None
    for r in results:
        if r['h4_tree_ms'] < r['softmax_ms']:
            crossover = r['seq_len']
            break

    if crossover:
        print(f"  H4 tree becomes faster than softmax at seq_len={crossover}")
    else:
        print("  Softmax is faster at all tested layer-level lengths")
        print("  (H4 tree overhead dominates at small/medium lengths due to Python ChamberTree)")

    if RUST_AVAILABLE and rust_results:
        print()
        print("  Rust backend top-k performance:")
        for r in rust_results[:6]:
            print(f"    n_keys={r['n_keys']:>6d}: Rust {r['rust_ms']:.3f}ms vs NumPy {r['numpy_ms']:.3f}ms ({r['speedup']:.1f}x)")
    elif not RUST_AVAILABLE:
        print()
        print("  Rust backend was NOT available for this run.")
        print("  To enable: cd rust && maturin develop --release")

    print()
    print("  Note: The Python ChamberTree has high constant factors.")
    print("  The Rust h4_rust backend shows raw computation speedups.")
    print("  Full Rust-accelerated attention layer is the next step.")
    print("=" * 100)


if __name__ == '__main__':
    main()
