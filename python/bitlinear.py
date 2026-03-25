"""
BitLinear: ternary {-1, 0, +1} linear layer with straight-through estimator.

Training: shadow float weights -> quantize forward -> STE backward
Inference: pure ternary weights -> matmul is add/sub only

Based on BitNet b1.58 (arxiv 2402.17764).

Drop-in replacement for nn.Linear. Use `use_bitlinear=True` in H4AttentionLayer
and H4TransformerBlock to swap all trainable projections to ternary.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def ternary_quantize(w):
    """Quantize weights to {-1, 0, +1} via absmean scaling.

    scale = mean(|w|)
    w_q = RoundClip(w / scale, -1, +1)

    The absmean adapts the rounding boundary to each layer's weight
    distribution. This is the canonical BitNet b1.58 method.
    """
    scale = w.abs().mean() + 1e-8
    w_scaled = w / scale
    w_q = torch.clamp(torch.round(w_scaled), -1, 1)
    return w_q, scale


def activation_quant_int8(x):
    """Per-token absmax quantization to int8 range [-127, 127].

    Each token (last dim) gets its own scale factor.
    """
    Q_b = 127.0
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    x_q = torch.clamp(torch.round(x * Q_b / scale), -Q_b, Q_b)
    return x_q, scale, Q_b


class BitLinear(nn.Module):
    """
    Ternary linear layer. Drop-in replacement for nn.Linear.

    Forward pass uses quantized weights via STE so gradients
    flow to shadow float weights. Inference mode freezes to
    pure ternary for integer-only compute.
    """

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        # Kaiming init scaled for ternary convergence
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.register_buffer('_frozen_ternary', None)
        self.register_buffer('_frozen_scale', None)
        self._inference_mode = False

    def forward(self, x):
        if self._inference_mode and self._frozen_ternary is not None:
            # Pure integer inference path
            y = F.linear(x, self._frozen_ternary.float() * self._frozen_scale, self.bias)
            return y

        # QAT forward with straight-through estimator (STE)
        #
        # Weight STE: forward sees quantized weights, backward sees float shadow
        w_q, w_scale = ternary_quantize(self.weight)
        w_ste = self.weight + (w_q * w_scale - self.weight).detach()

        # Activation STE: forward sees int8-quantized input, backward sees float
        x_q, x_scale, Q_b = activation_quant_int8(x)
        x_ste = x + (x_q * x_scale / Q_b - x).detach()

        # Matmul through STE — gradients flow to self.weight and x
        y = F.linear(x_ste, w_ste, self.bias)
        return y

    def freeze(self):
        """Lock to ternary for inference. After this, forward uses int path."""
        w_q, w_s = ternary_quantize(self.weight.data)
        self._frozen_ternary = w_q.to(torch.int8)
        self._frozen_scale = w_s
        self._inference_mode = True

    def unfreeze(self):
        """Return to training mode with float shadow weights."""
        self._inference_mode = False

    @property
    def ternary_stats(self):
        """Distribution of {-1, 0, +1} in current ternary quantization."""
        w_q, _ = ternary_quantize(self.weight.data)
        n = w_q.numel()
        return {
            'neg1': (w_q == -1).sum().item() / n,
            'zero': (w_q == 0).sum().item() / n,
            'pos1': (w_q == 1).sum().item() / n,
        }

    def extra_repr(self):
        s = f'{self.in_features}, {self.out_features}, bias={self.bias is not None}'
        if self._inference_mode:
            s += ', frozen=True'
        return s
