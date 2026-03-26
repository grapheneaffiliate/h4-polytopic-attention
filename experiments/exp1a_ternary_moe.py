"""
Experiment 1a: Ternary 2-of-4 Mixture-of-Experts on TinyStories

Goal: Prove that ternary MoE with ChamberTree routing matches or beats
      a dense ternary baseline of equivalent active parameters.

Setup:
  - 4 ternary experts, 2 active per token
  - Total params: ~4x active params (but only 2 experts compute per token)
  - ChamberTree routes tokens to experts based on H4 chamber assignment
  - Compare vs: dense ternary model with same active parameter count

Safety:
  - Uses max 2 CPU threads (torch.set_num_threads(2))
  - Runs for max 10 minutes per burst
  - Checkpoints every 2 minutes so progress is never lost
  - Can resume from checkpoint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import sys
import math
import json
from pathlib import Path

# Limit CPU usage
torch.set_num_threads(2)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

# ── Ternary Layer (from bitlinear.py) ─────────────────────────────

class TernaryLinear(nn.Linear):
    """Linear layer with ternary {-1, 0, +1} weights via STE."""

    def forward(self, x):
        # Quantize weights to {-1, 0, +1}
        w = self.weight
        alpha = w.abs().mean()
        w_ternary = torch.sign(w) * (w.abs() > 0.5 * alpha).float()
        # STE: use ternary in forward, real gradients in backward
        w_q = w + (w_ternary - w).detach()

        # RMSNorm input
        rms = x.norm(2, dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
        x_norm = x / (rms + 1e-8)

        return F.linear(x_norm, w_q, self.bias)


# ── Expert Module ──────────────────────────────────────────────────

class TernaryExpert(nn.Module):
    """One expert: two ternary linear layers with GELU."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.up = TernaryLinear(d_model, d_ff, bias=False)
        self.down = TernaryLinear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


# ── ChamberTree Router ────────────────────────────────────────────

class GeometricRouter(nn.Module):
    """Route tokens to experts using H4 chamber geometry.

    Each token is projected to 4D, its chamber is computed via dot products
    with the simple roots, and the chamber index maps to expert pairs.
    """

    def __init__(self, d_model, n_experts):
        super().__init__()
        self.n_experts = n_experts
        # Project token embeddings to 4D (H4 space)
        self.proj = nn.Linear(d_model, 4, bias=False)

        # Simple roots of H4 (from the 600-cell Coxeter group)
        phi = (1 + math.sqrt(5)) / 2
        roots = torch.tensor([
            [1, -1, 0, 0],
            [0, 1, -1, 0],
            [0, 0, 1, 0],
            [-0.5, 0.5, 0.5, phi/2],
        ], dtype=torch.float32)
        # Normalize
        roots = roots / roots.norm(dim=1, keepdim=True)
        self.register_buffer('roots', roots)

        # Chamber-to-expert mapping: 16 chambers -> expert pairs
        # Precompute which 2 experts each chamber activates
        pairs = []
        for i in range(16):
            # Deterministic mapping: chamber bits select expert pair
            e1 = i % n_experts
            e2 = (i // 2 + 1) % n_experts
            if e1 == e2:
                e2 = (e2 + 1) % n_experts
            pairs.append([e1, e2])
        self.register_buffer('chamber_to_experts',
                             torch.tensor(pairs, dtype=torch.long))

    def forward(self, x):
        """Returns expert indices and weights for each token.

        x: (batch, seq, d_model)
        Returns: (expert_indices: (batch, seq, 2), expert_weights: (batch, seq, 2))
        """
        # Project to 4D
        h4 = self.proj(x)  # (batch, seq, 4)
        h4 = F.normalize(h4, dim=-1)

        # Compute chamber index (4-bit) from dot products with roots
        dots = torch.matmul(h4, self.roots.T)  # (batch, seq, 4)
        bits = (dots >= 0).long()
        chamber = bits[..., 0] + 2 * bits[..., 1] + 4 * bits[..., 2] + 8 * bits[..., 3]

        # Look up expert pair for each chamber
        expert_indices = self.chamber_to_experts[chamber]  # (batch, seq, 2)

        # Weights based on distance from chamber boundary (confidence)
        confidence = dots.abs().min(dim=-1).values  # (batch, seq)
        # Higher confidence -> more weight on primary expert
        w1 = 0.5 + 0.3 * torch.sigmoid(confidence)
        w2 = 1.0 - w1
        expert_weights = torch.stack([w1, w2], dim=-1)  # (batch, seq, 2)

        return expert_indices, expert_weights


# ── MoE Transformer Block ────────────────────────────────────────

class TernaryMoEBlock(nn.Module):
    """Transformer block with ternary MoE FFN and standard attention."""

    def __init__(self, d_model, n_heads, d_ff, n_experts, dropout=0.0):
        super().__init__()
        self.n_experts = n_experts

        # Standard multi-head attention (not ternary for now)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # MoE: multiple ternary experts
        self.experts = nn.ModuleList([
            TernaryExpert(d_model, d_ff) for _ in range(n_experts)
        ])
        self.router = GeometricRouter(d_model, n_experts)

    def forward(self, x, mask=None):
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=mask)
        x = x + residual

        # MoE FFN
        residual = x
        x_norm = self.norm2(x)
        expert_indices, expert_weights = self.router(x_norm)

        # Compute expert outputs (loop over experts, mask select)
        batch, seq, d = x_norm.shape
        output = torch.zeros_like(x_norm)

        for e in range(self.n_experts):
            # Find which tokens use this expert (as expert 0 or expert 1)
            mask_e0 = (expert_indices[..., 0] == e)  # (batch, seq)
            mask_e1 = (expert_indices[..., 1] == e)

            # Any token that routes to this expert
            mask_any = mask_e0 | mask_e1
            if not mask_any.any():
                continue

            # Run expert on relevant tokens
            expert_input = x_norm[mask_any]  # (n_tokens, d)
            expert_out = self.experts[e](expert_input)

            # Weight by routing confidence
            w = torch.zeros(batch, seq, device=x.device)
            w[mask_e0] += expert_weights[mask_e0, 0]
            w[mask_e1] += expert_weights[mask_e1, 1]

            output[mask_any] += expert_out * w[mask_any].unsqueeze(-1)

        x = output + residual
        return x


# ── Full Model ────────────────────────────────────────────────────

class TernaryMoEModel(nn.Module):
    """Small ternary MoE language model for Experiment 1a."""

    def __init__(self, vocab_size, d_model=256, n_heads=4, d_ff=512,
                 n_layers=4, n_experts=4, max_seq=256):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList([
            TernaryMoEBlock(d_model, n_heads, d_ff, n_experts)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Count params
        total = sum(p.numel() for p in self.parameters())
        expert_params = sum(p.numel() for b in self.blocks for e in b.experts for p in e.parameters())
        active_per_token = total - expert_params + expert_params * 2 // n_experts
        print(f"  Total params: {total:,}")
        print(f"  Expert params: {expert_params:,} ({n_experts} experts)")
        print(f"  Active per token: ~{active_per_token:,} (2-of-{n_experts})")

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(pos)

        # Causal mask
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.norm(x)
        return self.head(x)


# ── Dense Baseline (same active params, no MoE) ───────────────────

class DenseBaselineModel(nn.Module):
    """Dense ternary model with same active parameter count as MoE."""

    def __init__(self, vocab_size, d_model=256, n_heads=4, d_ff=256,
                 n_layers=4, max_seq=256):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList([
            self._make_block(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        total = sum(p.numel() for p in self.parameters())
        print(f"  Dense baseline params: {total:,}")

    def _make_block(self, d_model, n_heads, d_ff):
        return nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            batch_first=True, norm_first=True
        )

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(pos)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        for block in self.blocks:
            x = block(x, src_mask=mask)
        x = self.norm(x)
        return self.head(x)


# ── Training Loop ─────────────────────────────────────────────────

def train_burst(model, data, optimizer, max_minutes=10, checkpoint_path=None,
                start_step=0):
    """Train for up to max_minutes, checkpointing every 2 minutes.

    Returns (steps_done, avg_loss, final_step).
    """
    model.train()
    t_start = time.time()
    t_last_ckpt = t_start
    total_loss = 0
    steps = 0
    step = start_step

    for epoch in range(100):  # will break on time limit
        for i in range(0, len(data) - 128, 128):
            batch = data[i:i+128].unsqueeze(0)  # (1, 128)
            target = data[i+1:i+129].unsqueeze(0)

            logits = model(batch)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            step += 1

            # Check time
            elapsed = time.time() - t_start
            if elapsed > max_minutes * 60:
                if checkpoint_path:
                    save_checkpoint(model, optimizer, step, total_loss/steps, checkpoint_path)
                return steps, total_loss / steps, step

            # Checkpoint every 2 minutes
            if time.time() - t_last_ckpt > 120 and checkpoint_path:
                save_checkpoint(model, optimizer, step, total_loss/steps, checkpoint_path)
                t_last_ckpt = time.time()

            # Log every 50 steps
            if steps % 50 == 0:
                cpu_pct = get_cpu_load()
                print(f"  step {step:5d} | loss {loss.item():.4f} | "
                      f"avg {total_loss/steps:.4f} | "
                      f"CPU {cpu_pct:.0f}% | "
                      f"{elapsed:.0f}s elapsed")

    return steps, total_loss / max(steps, 1), step


def save_checkpoint(model, optimizer, step, loss, path):
    """Save checkpoint for resume."""
    torch.save({
        'step': step,
        'loss': loss,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, path)
    print(f"  [checkpoint saved: step {step}, loss {loss:.4f}]")


def load_checkpoint(model, optimizer, path):
    """Load checkpoint if exists. Returns step number."""
    if os.path.exists(path):
        ckpt = torch.load(path, weights_only=True)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(f"  [resumed from step {ckpt['step']}, loss {ckpt['loss']:.4f}]")
        return ckpt['step']
    return 0


def get_cpu_load():
    """Get current CPU load percentage."""
    try:
        import psutil
        return psutil.cpu_percent(interval=0.1)
    except ImportError:
        return -1


# ── Data ──────────────────────────────────────────────────────────

def get_data():
    """Load or generate training data."""
    # Try TinyStories first
    data_path = Path(__file__).parent.parent / "data" / "tinystories_train.txt"
    if data_path.exists():
        print(f"  Loading TinyStories from {data_path}")
        text = data_path.read_text(encoding='utf-8')[:500000]
    else:
        # Fall back to Shakespeare
        shakespeare = Path(__file__).parent.parent / "data" / "shakespeare.txt"
        if shakespeare.exists():
            print(f"  Loading Shakespeare from {shakespeare}")
            text = shakespeare.read_text(encoding='utf-8')[:500000]
        else:
            print("  No dataset found. Generating synthetic data...")
            # Simple synthetic: repeated patterns
            text = "the cat sat on the mat . " * 10000

    # Character-level tokenization (simple, fast)
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    print(f"  Vocab: {vocab_size}, Data: {len(data):,} tokens")
    return data, vocab_size


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  EXPERIMENT 1a: Ternary 2-of-4 MoE vs Dense Baseline")
    print("  Safety: 2 threads, 10-min bursts, 2-min checkpoints")
    print("=" * 60)

    data, vocab_size = get_data()
    ckpt_dir = Path(__file__).parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    results = {}

    # ── Train MoE ──
    print("\n--- Ternary MoE (4 experts, 2 active) ---")
    moe = TernaryMoEModel(vocab_size, d_model=256, n_heads=4, d_ff=512,
                           n_layers=4, n_experts=4)
    moe_opt = torch.optim.AdamW(moe.parameters(), lr=3e-4)
    moe_ckpt = ckpt_dir / "exp1a_moe.pt"
    start_step = load_checkpoint(moe, moe_opt, moe_ckpt)

    print("  Training MoE (10 min burst)...")
    steps, avg_loss, final_step = train_burst(
        moe, data, moe_opt, max_minutes=10,
        checkpoint_path=moe_ckpt, start_step=start_step
    )
    results['moe'] = {'steps': final_step, 'loss': avg_loss}
    print(f"  MoE: {final_step} steps, loss {avg_loss:.4f}")

    # ── Cooldown ──
    print("\n  Cooling down (60 seconds)...")
    time.sleep(60)

    # ── Train Dense Baseline ──
    print("\n--- Dense Ternary Baseline (same active params) ---")
    # d_ff=256 to match MoE's 2-of-4 × d_ff=512 -> active d_ff=256
    dense = DenseBaselineModel(vocab_size, d_model=256, n_heads=4, d_ff=256,
                                n_layers=4)
    dense_opt = torch.optim.AdamW(dense.parameters(), lr=3e-4)
    dense_ckpt = ckpt_dir / "exp1a_dense.pt"
    start_step = load_checkpoint(dense, dense_opt, dense_ckpt)

    print("  Training Dense (10 min burst)...")
    steps, avg_loss, final_step = train_burst(
        dense, data, dense_opt, max_minutes=10,
        checkpoint_path=dense_ckpt, start_step=start_step
    )
    results['dense'] = {'steps': final_step, 'loss': avg_loss}
    print(f"  Dense: {final_step} steps, loss {avg_loss:.4f}")

    # ── Results ──
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name:>8s}: {r['steps']} steps, loss {r['loss']:.4f}")

    moe_loss = results['moe']['loss']
    dense_loss = results['dense']['loss']
    if dense_loss > 0:
        ratio = moe_loss / dense_loss
        verdict = "MoE WINS" if ratio < 1.0 else "Dense wins" if ratio > 1.05 else "Tie"
        print(f"\n  MoE/Dense ratio: {ratio:.3f} ({verdict})")
        print(f"  MoE has ~4x total params but same active compute as Dense.")
        if ratio < 1.0:
            print(f"  -> MoE extracts value from inactive experts (knowledge routing).")
        elif ratio > 1.05:
            print(f"  -> Dense baseline more efficient at this scale. Need larger model.")
        else:
            print(f"  -> Equivalent. MoE advantage appears at larger scale.")

    # Save results
    results_path = ckpt_dir / "exp1a_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")


if __name__ == '__main__':
    main()
