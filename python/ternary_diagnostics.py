"""
Diagnostics for ternary quantization x H4 geometry interaction.

Key questions answered:
1. Does ternary W_nudge preserve chamber navigation?
2. What's the weight distribution structure?
3. Does the rank-1 pattern from Phase 5 survive ternary?
4. How does geo_alignment change: float vs ternary?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bitlinear import ternary_quantize, BitLinear
from h4_polytopic_attention import generate_600_cell_vertices, build_coxeter_chambers


def chamber_preservation(model, n_test=10000):
    """
    Measure how often float vs ternary nudge assigns the same chamber.

    Generates random queries on S^3, applies float and ternary nudge,
    computes chamber indices via sign(dot(q, root_i)), reports agreement.

    Phase 5 showed 96.5% geo alignment with float nudge.
    Theory predicts >90% preservation under ternary because
    chambers only depend on signs, and ternary preserves signs.
    """
    # Get simple roots from the model's first attention layer
    attn = _find_h4_attention(model)
    if attn is None:
        print("  No H4AttentionLayer found in model")
        return {}

    roots = attn.simple_roots  # (4, 4)
    results = {}

    for h in range(attn.n_heads):
        # Generate random unit queries on S^3
        q = torch.randn(n_test, 4)
        q = F.normalize(q, dim=-1)

        # Float nudge path
        W_float = attn.W_nudge[h].detach()  # (4, 4)
        with torch.no_grad():
            q_float = (q @ W_float.T)
            q_float = F.normalize(q_float, dim=-1)
            ch_float = (q_float @ roots.T >= 0).int()  # (n_test, 4)

        # Ternary nudge path
        w_q, w_s = ternary_quantize(W_float)
        W_tern = w_q * w_s
        with torch.no_grad():
            q_tern = (q @ W_tern.T)
            q_tern = F.normalize(q_tern, dim=-1)
            ch_tern = (q_tern @ roots.T >= 0).int()

        # Agreement: all 4 sign bits match
        agree = (ch_float == ch_tern).all(dim=-1).float().mean().item()
        results[f'head_{h}'] = agree

    return results


def nudge_ternary_structure(model):
    """
    Analyze ternary W_nudge: SVD, zero pattern, 600-cell alignment.

    Expected from Phase 5 observations (nudge_rank=1.68, geo_align=0.965):
    - Ternary SVD should show a dominant sigma_1
    - High zero% = head learned to ignore some Coxeter directions
    - Dominant direction should still align with 600-cell vertices
    """
    attn = _find_h4_attention(model)
    if attn is None:
        return {}

    verts = attn.vertices  # (120, 4)
    results = {}

    for h in range(attn.n_heads):
        W = attn.W_nudge[h].detach()
        delta = W - torch.eye(4, device=W.device)

        # Float SVD
        s_float = torch.linalg.svdvals(delta)

        # Ternary quantization
        w_q, w_s = ternary_quantize(W)
        w_tern = w_q.float()
        delta_tern = w_tern - torch.eye(4, device=w_tern.device) / w_s

        # Ternary SVD
        U, S_tern, Vt = torch.linalg.svd(w_tern)
        s_tern = S_tern

        # Ternary weight distribution
        n = w_q.numel()
        stats = {
            'neg1': (w_q == -1).sum().item() / n,
            'zero': (w_q == 0).sum().item() / n,
            'pos1': (w_q == 1).sum().item() / n,
        }

        # Dominant direction alignment with 600-cell
        dominant = Vt[0]
        alignment_tern = (dominant @ verts.T).abs().max().item()

        # Float dominant direction for comparison
        _, _, Vt_float = torch.linalg.svd(delta)
        dominant_float = Vt_float[0]
        alignment_float = (dominant_float @ verts.T).abs().max().item()

        # Rank ratio
        rank_ratio_tern = (s_tern[0] / s_tern[1]).item() if s_tern[1] > 1e-8 else float('inf')
        rank_ratio_float = (s_float[0] / s_float[1]).item() if s_float[1] > 1e-8 else float('inf')

        results[f'head_{h}'] = {
            'svd_float': s_float.tolist(),
            'svd_ternary': s_tern.tolist(),
            'rank_ratio_float': rank_ratio_float,
            'rank_ratio_ternary': rank_ratio_tern,
            'alignment_float': alignment_float,
            'alignment_ternary': alignment_tern,
            'weight_dist': stats,
            'zero_rows': (w_q == 0).all(dim=1).sum().item(),
            'zero_cols': (w_q == 0).all(dim=0).sum().item(),
        }

    return results


def bitlinear_layer_stats(model):
    """Collect ternary stats across all BitLinear layers in the model."""
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            s = module.ternary_stats
            stats[name] = {
                'shape': list(module.weight.shape),
                'params': module.weight.numel(),
                **s,
            }
    return stats


def size_comparison(model):
    """Compare model size: float32 vs ternary."""
    total_float_params = 0
    total_ternary_params = 0
    total_buffer_elems = 0

    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            total_ternary_params += module.weight.numel()
        elif isinstance(module, nn.Linear):
            total_float_params += module.weight.numel()
            if module.bias is not None:
                total_float_params += module.bias.numel()

    for buf in model.buffers():
        total_buffer_elems += buf.numel()

    # Other parameters (embeddings, layer norms, W_nudge, chamber_bonus)
    other_params = sum(p.numel() for p in model.parameters()) - total_ternary_params - total_float_params

    float32_bits = (total_float_params + total_ternary_params + other_params) * 32
    mixed_bits = total_ternary_params * 1.58 + (total_float_params + other_params) * 32
    buffer_bits = total_buffer_elems * 32

    return {
        'ternary_params': total_ternary_params,
        'float_params': total_float_params,
        'other_params': other_params,
        'buffer_elems': total_buffer_elems,
        'float32_kb': float32_bits / 8 / 1024,
        'mixed_kb': mixed_bits / 8 / 1024,
        'buffer_kb': buffer_bits / 8 / 1024,
        'compression': float32_bits / mixed_bits if mixed_bits > 0 else 1.0,
    }


def full_report(model):
    """Print full ternary diagnostic report."""
    print("=" * 64)
    print("  TERNARY DIAGNOSTICS: H4 x BitNet b1.58")
    print("=" * 64)

    # Check if model uses BitLinear
    has_bitlinear = any(isinstance(m, BitLinear) for m in model.modules())
    attn = _find_h4_attention(model)

    if attn is None:
        print("  No H4AttentionLayer found in model")
        return

    # Chamber preservation
    print("\n  Chamber Preservation (float vs ternary nudge):")
    print("  (Agreement rate: same Coxeter chamber under both paths)")
    cp = chamber_preservation(model)
    mean_cp = 0.0
    for head, agree in cp.items():
        status = "OK" if agree > 0.90 else "WARN" if agree > 0.80 else "FAIL"
        print(f"    {head}: {agree:.1%} [{status}]")
        mean_cp += agree
    if cp:
        mean_cp /= len(cp)
        print(f"    MEAN: {mean_cp:.1%}")

    # Nudge structure
    print("\n  Ternary Nudge Structure:")
    ns = nudge_ternary_structure(model)
    for head, info in ns.items():
        print(f"    {head}:")
        print(f"      Float SVD:   [{', '.join(f'{s:.3f}' for s in info['svd_float'])}]")
        print(f"      Ternary SVD: [{', '.join(f'{s:.3f}' for s in info['svd_ternary'])}]")
        rr_f = info['rank_ratio_float']
        rr_t = info['rank_ratio_ternary']
        print(f"      Rank ratio:  float={rr_f:.1f}x  ternary={rr_t:.1f}x")
        print(f"      600-cell alignment: float={info['alignment_float']:.3f}  ternary={info['alignment_ternary']:.3f}")
        d = info['weight_dist']
        print(f"      Ternary weights: -1={d['neg1']:.0%}  0={d['zero']:.0%}  +1={d['pos1']:.0%}")
        if info['zero_rows'] > 0 or info['zero_cols'] > 0:
            print(f"      Zero rows/cols: {info['zero_rows']}/4, {info['zero_cols']}/4")

    # BitLinear layer stats
    if has_bitlinear:
        print("\n  BitLinear Layers:")
        bl_stats = bitlinear_layer_stats(model)
        for name, s in bl_stats.items():
            print(f"    {name:40s}  {s['params']:6d} params  "
                  f"-1={s['neg1']:.0%} 0={s['zero']:.0%} +1={s['pos1']:.0%}")

    # Size comparison
    print("\n  Size Comparison:")
    sz = size_comparison(model)
    print(f"    Ternary params:  {sz['ternary_params']:,} ({sz['ternary_params'] * 1.58 / 8 / 1024:.1f} KB at 1.58 bits)")
    print(f"    Float params:    {sz['float_params']:,} ({sz['float_params'] * 4 / 1024:.1f} KB)")
    print(f"    Other params:    {sz['other_params']:,} ({sz['other_params'] * 4 / 1024:.1f} KB)")
    print(f"    Buffers:         {sz['buffer_elems']:,} ({sz['buffer_kb']:.1f} KB)")
    print(f"    All-float32:     {sz['float32_kb']:.1f} KB")
    print(f"    Mixed ternary:   {sz['mixed_kb']:.1f} KB")
    print(f"    Compression:     {sz['compression']:.1f}x")
    print("=" * 64)


def _find_h4_attention(model):
    """Find the first H4AttentionLayer in a model."""
    from h4_hybrid_attention import H4AttentionLayer
    for module in model.modules():
        if isinstance(module, H4AttentionLayer):
            return module
    return None


if __name__ == '__main__':
    from h4_language_model import H4LanguageModel

    print("Testing with float model...")
    model_float = H4LanguageModel(vocab_size=128, d_model=64, n_heads=8, n_layers=2, use_bitlinear=False)
    full_report(model_float)

    print("\n\nTesting with BitLinear model...")
    model_tern = H4LanguageModel(vocab_size=128, d_model=64, n_heads=8, n_layers=2, use_bitlinear=True)
    full_report(model_tern)
