"""
Progressive H4 attention swap for pre-trained models.

Takes a standard HuggingFace transformer model and gradually replaces
softmax attention with H4 polytopic attention via a gated adapter.

The swap happens in 4 phases:
1. Adapter: H4 as parallel path with learned gate (starts at 0)
2. Hybrid: both pathways train together, gate opens
3. Selective swap: layers with gate > threshold keep only H4
4. Ternary quantization: BitLinear on H4 layers

SmolLM3-3B uses GQA with 4 groups — maps naturally to H4's 4 Coxeter roots.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from h4_hybrid_attention import H4AttentionLayer
from bitlinear import BitLinear


class H4AttentionAdapter(nn.Module):
    """
    Wraps an existing attention layer with a parallel H4 path.

    output = (1 - gate) * original_attention(x) + gate * h4_attention(x)

    Gate starts at ~0 (sigmoid(-5) ~ 0.007) so the original model
    is preserved. As training progresses, gate opens and H4 takes over.
    """

    def __init__(self, original_attention, d_model, n_heads, use_bitlinear=True):
        super().__init__()
        self.original = original_attention
        self.h4 = H4AttentionLayer(
            d_model=d_model,
            n_heads=n_heads,
            d_value=d_model // n_heads,
            use_bitlinear=use_bitlinear,
        )
        # Gate parameter: starts at -5 so sigmoid(-5) ~ 0.007
        self.gate_logit = nn.Parameter(torch.tensor(-5.0))

    @property
    def gate(self):
        return torch.sigmoid(self.gate_logit)

    def forward(self, hidden_states, **kwargs):
        # Original attention path
        orig_out = self.original(hidden_states, **kwargs)
        # Handle different return types (some return tuples)
        if isinstance(orig_out, tuple):
            orig_out = orig_out[0]

        # H4 attention path
        h4_out = self.h4(hidden_states, use_tree=False)

        # Gated combination
        g = self.gate
        return (1 - g) * orig_out + g * h4_out


def swap_attention_layers(model, d_model, n_heads, use_bitlinear=True):
    """
    Replace all attention layers in a HuggingFace model with H4 adapters.

    Works with SmolLM3, SmolLM2, OLMo, TinyLlama, Qwen, and most
    LLaMA-style models. Finds attention modules by looking for
    q_proj/k_proj/v_proj patterns.

    Returns the model with adapters installed and count of swapped layers.
    """
    swapped = 0

    for name, module in list(model.named_modules()):
        # Find self-attention modules (different models use different names)
        is_attention = (
            hasattr(module, 'q_proj') and
            hasattr(module, 'k_proj') and
            hasattr(module, 'v_proj')
        )

        if is_attention:
            # Navigate to parent to replace the module
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            child_name = parts[-1]

            adapter = H4AttentionAdapter(
                module, d_model, n_heads, use_bitlinear
            )
            setattr(parent, child_name, adapter)
            swapped += 1

    return model, swapped


def get_gate_values(model):
    """Report gate values across all adapted layers."""
    gates = {}
    for name, module in model.named_modules():
        if isinstance(module, H4AttentionAdapter):
            gates[name] = module.gate.item()
    return gates


def freeze_original_attention(model):
    """Freeze original attention weights (only H4 adapter trains)."""
    for module in model.modules():
        if isinstance(module, H4AttentionAdapter):
            for param in module.original.parameters():
                param.requires_grad = False


def unfreeze_original_attention(model):
    """Unfreeze original attention for hybrid training."""
    for module in model.modules():
        if isinstance(module, H4AttentionAdapter):
            for param in module.original.parameters():
                param.requires_grad = True


def selective_swap(model, high_threshold=0.8, low_threshold=0.3):
    """
    Phase 3: Replace adapters with final architecture.

    - Layers with gate > high_threshold: keep only H4, remove original
    - Layers with gate < low_threshold: keep only original, remove H4
    - Layers in between: keep hybrid adapter

    Returns dict with swap decisions per layer.
    """
    decisions = {}
    for name, module in list(model.named_modules()):
        if isinstance(module, H4AttentionAdapter):
            gate = module.gate.item()
            if gate > high_threshold:
                decisions[name] = ('h4_only', gate)
            elif gate < low_threshold:
                decisions[name] = ('original_only', gate)
            else:
                decisions[name] = ('hybrid', gate)

    return decisions


def count_params(model):
    """Count parameters by category."""
    h4_params = 0
    gate_params = 0
    original_params = 0
    other_params = 0

    for name, param in model.named_parameters():
        n = param.numel()
        if 'h4' in name:
            h4_params += n
        elif 'gate' in name:
            gate_params += n
        elif any(attn in name for attn in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            original_params += n
        else:
            other_params += n

    return {
        'h4': h4_params,
        'gate': gate_params,
        'original_attention': original_params,
        'other': other_params,
        'total': h4_params + gate_params + original_params + other_params,
    }
