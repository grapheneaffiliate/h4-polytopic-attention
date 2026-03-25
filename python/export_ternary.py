"""
Export trained H4 BitLinear model for deployment.

Freezes all BitLinear layers to pure ternary, packs weights,
saves alongside frozen geometric constants.

Output format: PyTorch state dict with:
  - geometry.* : frozen H4/E8 buffers (float32)
  - ternary.*.weight : int8 with values {-1, 0, +1}
  - ternary.*.scale : float32 per-layer scale
  - params.* : remaining float parameters (embeddings, norms, chamber_bonus)
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bitlinear import BitLinear


def export(model, path):
    """Export model with frozen ternary weights."""

    # Freeze all BitLinear layers
    frozen_count = 0
    for module in model.modules():
        if isinstance(module, BitLinear):
            module.freeze()
            frozen_count += 1

    # Collect all tensors
    state = {}

    # Frozen geometry (float32, static lookup tables)
    for name, buf in model.named_buffers():
        state[f'geometry.{name}'] = buf.cpu()

    # All parameters (float shadow weights, embeddings, norms, etc.)
    for name, param in model.named_parameters():
        state[f'params.{name}'] = param.data.cpu()

    # Frozen ternary buffers (int8 packed)
    for name, module in model.named_modules():
        if isinstance(module, BitLinear) and module._frozen_ternary is not None:
            state[f'ternary.{name}.weight'] = module._frozen_ternary.cpu()
            state[f'ternary.{name}.scale'] = module._frozen_scale.cpu()

    torch.save(state, path)

    # Report
    ternary_params = sum(
        t.numel() for k, t in state.items()
        if k.startswith('ternary.') and 'scale' not in k
    )
    geom_elems = sum(
        t.numel() for k, t in state.items()
        if k.startswith('geometry.')
    )
    learned_params = sum(
        t.numel() for k, t in state.items()
        if k.startswith('params.')
    )

    ternary_kb = ternary_params * 1.58 / 8 / 1024
    geom_kb = geom_elems * 4 / 1024
    learned_kb = learned_params * 4 / 1024
    total_mixed = ternary_kb + geom_kb + learned_kb
    total_float = (ternary_params + geom_elems + learned_params) * 4 / 1024

    print(f"Exported to {path}")
    print(f"  Frozen BitLinear layers: {frozen_count}")
    print(f"  Ternary weights: {ternary_params:,} ({ternary_kb:.1f} KB at 1.58 bits)")
    print(f"  Geometry buffers: {geom_elems:,} ({geom_kb:.1f} KB float32)")
    print(f"  Learned float:   {learned_params:,} ({learned_kb:.1f} KB float32)")
    print(f"  Total mixed:     {total_mixed:.1f} KB")
    print(f"  vs all-float32:  {total_float:.1f} KB")
    print(f"  Compression:     {total_float / total_mixed:.1f}x")


if __name__ == '__main__':
    from h4_language_model import H4LanguageModel

    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    out = sys.argv[2] if len(sys.argv) > 2 else 'h4_ternary_model.pt'

    # Default config for demo
    model = H4LanguageModel(
        vocab_size=256,
        d_model=64,
        n_heads=8,
        n_layers=2,
        d_value=16,
        use_bitlinear=True,
    )

    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        print(f"Loaded checkpoint from {ckpt}")
    else:
        print("No checkpoint provided, exporting untrained model (for format verification)")

    export(model, out)
