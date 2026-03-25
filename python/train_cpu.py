"""
H4 Polytopic Attention — CPU autoresearch training script.
This is the ONLY file the agent modifies during autonomous research.

Follows the autoresearch pattern: modify → run (2 min budget) → measure → keep/discard.

The frozen H4 geometry is off-limits. Only the trainable adapters, hyperparameters,
training loop details, and architecture of trainable layers may be changed.
"""

import os
import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h4_polytopic_attention import generate_600_cell_vertices, build_coxeter_chambers
from h4_language_model import H4LanguageModel
from bitlinear import BitLinear

# ---------------------------------------------------------------------------
# Constants (DO NOT MODIFY the frozen geometry section)
# ---------------------------------------------------------------------------

PHI = (1 + math.sqrt(5)) / 2

# Frozen geometric constants — loaded from existing code
VERTICES = torch.tensor(generate_600_cell_vertices(), dtype=torch.float32)
CHAMBERS = build_coxeter_chambers(VERTICES.numpy())
SIMPLE_ROOTS = torch.tensor(CHAMBERS['simple_roots'], dtype=torch.float32)

# ---------------------------------------------------------------------------
# Hyperparameters (AGENT MAY MODIFY THESE)
# ---------------------------------------------------------------------------

# Time budget: 2 minutes on CPU
TIME_BUDGET = 120  # seconds

# Dataset: 'synthetic', 'shakespeare', or 'tinystories'
DATASET = 'synthetic'

# Data
MAX_SEQ_LEN = 128
BATCH_SIZE = 8

# Model
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 4
D_VALUE = 16
D_FFN = 512
TOP_K = 16
DROPOUT = 0.0
USE_BITLINEAR = True  # Set True for ternary {-1,0,+1} weights

# Optimizer
LR = 5e-3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 50
GRAD_CLIP = 1.0

# Eval
EVAL_INTERVAL = 25
EVAL_BATCHES = 5

# ---------------------------------------------------------------------------
# Data: Character-level Shakespeare (or synthetic if not available)
# ---------------------------------------------------------------------------

def load_text_data():
    """Load training text. Falls back to synthetic data if no file available."""
    # Try to load Shakespeare or other text
    data_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'data', 'shakespeare.txt'),
        os.path.join(os.path.dirname(__file__), '..', 'data', 'input.txt'),
        os.path.join(os.path.dirname(__file__), 'data', 'input.txt'),
    ]

    text = None
    for path in data_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"Loaded data from {path} ({len(text)} chars)")
            break

    if text is None:
        # Generate synthetic text with mathematical structure
        # Fibonacci-structured repetitions to test geometric inductive bias
        print("No data file found, generating synthetic text...")
        base_phrases = [
            "the golden ratio appears in nature ",
            "fibonacci numbers grow exponentially ",
            "symmetry underlies all of physics ",
            "the icosahedron has twenty faces ",
            "phi equals one plus one over phi ",
            "geometry is the language of space ",
            "five fold symmetry cannot tile a plane ",
            "the dodecahedron has twelve faces ",
        ]
        # Build text with Fibonacci-structured repetitions
        text = ""
        a, b = 1, 1
        for _ in range(200):
            phrase = base_phrases[a % len(base_phrases)]
            text += phrase * (b % 3 + 1)
            a, b = b, a + b

    return text


def prepare_char_dataset(text: str):
    """Prepare character-level dataset from text."""
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    # Split 90/10
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, vocab_size, stoi, itos


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int):
    """Sample a random batch of sequences."""
    max_start = len(data) - seq_len - 1
    if max_start <= 0:
        max_start = 1
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
    return x, y


# ---------------------------------------------------------------------------
# Training loop (follows autoresearch pattern)
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    if DATASET != 'synthetic':
        from prepare_data import load_and_prepare
        train_data, val_data, vocab_size, stoi, itos = load_and_prepare(DATASET)
    else:
        text = load_text_data()
        train_data, val_data, vocab_size, stoi, itos = prepare_char_dataset(text)
    print(f"Vocab size: {vocab_size}, Train: {len(train_data)}, Val: {len(val_data)}")

    # Create model
    model = H4LanguageModel(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_value=D_VALUE,
        d_ffn=D_FFN,
        top_k=TOP_K,
        max_seq_len=MAX_SEQ_LEN * 2,
        dropout=DROPOUT,
        use_bitlinear=USE_BITLINEAR,
    )

    param_info = model.count_params()
    print(f"Model params: {param_info['trainable']:,} trainable, {param_info['buffers']:,} buffer elements")

    # Optimizer: AdamW with cosine schedule
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    # Cosine LR schedule with warmup
    def lr_schedule(step):
        if step < WARMUP_STEPS:
            return step / max(WARMUP_STEPS, 1)
        # Cosine decay to 10% of peak
        progress = (step - WARMUP_STEPS) / max(1, 500 - WARMUP_STEPS)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Training state
    step = 0
    total_training_time = 0.0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # Use full attention (no tree) for short sequences during training
    # Tree is beneficial for long sequences; for seq_len=128, full attention is faster
    use_tree = MAX_SEQ_LEN > 256

    print(f"\nTraining for {TIME_BUDGET}s budget, seq_len={MAX_SEQ_LEN}, use_tree={use_tree}")
    print(f"{'step':>6} {'loss':>8} {'val_loss':>8} {'lr':>10} {'dt':>6} {'progress':>8}")
    print("-" * 56)

    model.train()

    while True:
        t0 = time.time()

        # Get batch
        x, y = get_batch(train_data, BATCH_SIZE, MAX_SEQ_LEN)

        # Forward
        logits = model(x, use_tree=use_tree)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()
        scheduler.step()

        dt = time.time() - t0
        if step > 2:  # skip warmup steps for timing
            total_training_time += dt

        train_losses.append(loss.item())

        # Eval
        val_loss = None
        if step % EVAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                vl = []
                for _ in range(EVAL_BATCHES):
                    xv, yv = get_batch(val_data, BATCH_SIZE, MAX_SEQ_LEN)
                    vlogits = model(xv, use_tree=False)
                    vl.append(F.cross_entropy(vlogits.view(-1, vocab_size), yv.view(-1)).item())
                val_loss = sum(vl) / len(vl)
                val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

            current_lr = scheduler.get_last_lr()[0]
            progress = min(total_training_time / TIME_BUDGET, 1.0)
            print(f"{step:6d} {loss.item():8.4f} {val_loss:8.4f} {current_lr:10.6f} {dt:6.3f} {progress:7.1%}")
            model.train()

        step += 1
        if step > 2 and total_training_time >= TIME_BUDGET:
            break

    # ---------------------------------------------------------------------------
    # Final evaluation
    # ---------------------------------------------------------------------------

    model.eval()
    with torch.no_grad():
        # Final val loss
        vl = []
        for _ in range(EVAL_BATCHES * 4):
            xv, yv = get_batch(val_data, BATCH_SIZE, MAX_SEQ_LEN)
            vlogits = model(xv, use_tree=False)
            vl.append(F.cross_entropy(vlogits.view(-1, vocab_size), yv.view(-1)).item())
        final_val_loss = sum(vl) / len(vl)

        # Bits per byte (for character-level: loss_nats / ln(2))
        val_bpb = final_val_loss / math.log(2)

        # Geometric diagnostics on a sample batch
        xd, _ = get_batch(val_data, 1, MAX_SEQ_LEN)
        _, diag_list = model(xd, use_tree=False, return_diagnostics=True)

        # Aggregate diagnostics across layers
        avg_chamber_entropy = np.mean([d['chamber_entropy'] for d in diag_list])
        nudge_ranks = []
        geo_aligns = []
        for d in diag_list:
            nudge_ranks.extend(d['nudge_rank'])
            geo_aligns.extend(d['geo_alignment'])
        avg_nudge_rank = np.mean([r for r in nudge_ranks if r != float('inf')] or [0])
        avg_geo_alignment = np.mean(geo_aligns)

        # Generate sample text
        seed_text = list(stoi.keys())[:4]  # first 4 chars
        seed_ids = torch.tensor([[stoi[c] for c in seed_text]], dtype=torch.long)
        generated = model.generate(seed_ids, max_new_tokens=80, temperature=0.8, top_k_sample=10)
        gen_text = ''.join([itos.get(i.item(), '?') for i in generated[0]])

    # ---------------------------------------------------------------------------
    # Summary (autoresearch-parseable format)
    # ---------------------------------------------------------------------------

    # Ternary diagnostics (if using BitLinear)
    has_bitlinear = any(isinstance(m, BitLinear) for m in model.modules())
    ternary_info = {}
    if has_bitlinear:
        from ternary_diagnostics import chamber_preservation, bitlinear_layer_stats, size_comparison
        cp = chamber_preservation(model)
        mean_cp = sum(cp.values()) / len(cp) if cp else 0.0
        bl_stats = bitlinear_layer_stats(model)
        mean_zero_pct = np.mean([s['zero'] for s in bl_stats.values()]) if bl_stats else 0.0
        sz = size_comparison(model)
        ternary_info = {
            'chamber_preserve': mean_cp,
            'mean_zero_pct': mean_zero_pct,
            'compression': sz['compression'],
            'mixed_kb': sz['mixed_kb'],
        }

    print("\n" + "=" * 60)
    print("GENERATED SAMPLE:")
    print(gen_text[:200])
    print("=" * 60)

    print("\n---")
    print(f"val_bpb:            {val_bpb:.6f}")
    print(f"val_loss:           {final_val_loss:.6f}")
    print(f"best_val_loss:      {best_val_loss:.6f}")
    print(f"chamber_entropy:    {avg_chamber_entropy:.4f}")
    print(f"avg_nudge_rank:     {avg_nudge_rank:.4f}")
    print(f"avg_geo_alignment:  {avg_geo_alignment:.4f}")
    print(f"training_seconds:   {total_training_time:.1f}")
    print(f"total_seconds:      {time.time() - t_start:.1f}")
    print(f"peak_memory_mb:     {0:.1f}")
    print(f"num_steps:          {step}")
    print(f"num_params:         {param_info['trainable']}")
    print(f"vocab_size:         {vocab_size}")
    print(f"seq_len:            {MAX_SEQ_LEN}")
    print(f"ternary:            {'yes' if USE_BITLINEAR else 'no'}")
    if ternary_info:
        print(f"chamber_preserve:   {ternary_info['chamber_preserve']:.4f}")
        print(f"mean_zero_pct:      {ternary_info['mean_zero_pct']:.4f}")
        print(f"compression:        {ternary_info['compression']:.1f}x")
        print(f"model_size_kb:      {ternary_info['mixed_kb']:.1f}")


if __name__ == '__main__':
    main()
