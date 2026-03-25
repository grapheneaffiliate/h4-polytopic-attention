"""
Head-to-head comparison: H4 attention vs softmax vs linear attention.
Same model size, same data, same training budget.

Usage:
    python compare_baselines.py                    # Shakespeare (default)
    python compare_baselines.py --dataset tinystories
    python compare_baselines.py --time-budget 60   # Faster runs
"""

import os
import sys
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prepare_data import load_and_prepare
from baselines import BaselineLanguageModel
from h4_language_model import H4LanguageModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Model architecture (same for all models)
D_MODEL = 128
N_HEADS = 8
N_LAYERS = 4
D_VALUE = 16
D_FFN = 512
MAX_SEQ_LEN = 128
DROPOUT = 0.0

# Training
BATCH_SIZE = 8
LR = 5e-3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 50
GRAD_CLIP = 1.0
TIME_BUDGET = 120  # seconds per model

# Eval
EVAL_INTERVAL = 25
EVAL_BATCHES = 5

# Models to compare
CONFIGS = [
    {'name': 'H4 Float',  'attention': 'h4',      'bitlinear': False},
    {'name': 'H4 Ternary', 'attention': 'h4',     'bitlinear': True},
    {'name': 'Softmax',    'attention': 'softmax', 'bitlinear': False},
    {'name': 'Linear',     'attention': 'linear',  'bitlinear': False},
]


def get_batch(data, batch_size, seq_len):
    """Sample a random batch of sequences."""
    max_start = len(data) - seq_len - 1
    if max_start <= 0:
        max_start = 1
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
    return x, y


def create_model(config, vocab_size):
    """Create a model based on config."""
    attn_type = config['attention']
    use_bitlinear = config['bitlinear']

    if attn_type == 'h4':
        model = H4LanguageModel(
            vocab_size=vocab_size,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            d_value=D_VALUE,
            d_ffn=D_FFN,
            top_k=16,
            max_seq_len=MAX_SEQ_LEN * 2,
            dropout=DROPOUT,
            use_bitlinear=use_bitlinear,
        )
    else:
        model = BaselineLanguageModel(
            vocab_size=vocab_size,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            d_value=D_VALUE,
            d_ffn=D_FFN,
            max_seq_len=MAX_SEQ_LEN * 2,
            dropout=DROPOUT,
            attention_type=attn_type,
            use_bitlinear=use_bitlinear,
        )
    return model


def train_and_evaluate(config, train_data, val_data, vocab_size, itos, time_budget):
    """Train a model and return evaluation metrics."""
    name = config['name']
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    np.random.seed(42)

    model = create_model(config, vocab_size)
    param_info = model.count_params()
    print(f"  Parameters: {param_info['trainable']:,} trainable")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    def lr_schedule(step):
        if step < WARMUP_STEPS:
            return step / max(WARMUP_STEPS, 1)
        progress = (step - WARMUP_STEPS) / max(1, 500 - WARMUP_STEPS)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # H4 models use full attention (no tree) for short sequences
    is_h4 = config['attention'] == 'h4'

    step = 0
    total_training_time = 0.0
    best_val_loss = float('inf')
    model.train()

    t_start = time.time()

    while True:
        t0 = time.time()

        x, y = get_batch(train_data, BATCH_SIZE, MAX_SEQ_LEN)

        if is_h4:
            logits = model(x, use_tree=False)
        else:
            logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        dt = time.time() - t0
        if step > 2:
            total_training_time += dt

        # Periodic eval
        if step % EVAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                vl = []
                for _ in range(EVAL_BATCHES):
                    xv, yv = get_batch(val_data, BATCH_SIZE, MAX_SEQ_LEN)
                    if is_h4:
                        vlogits = model(xv, use_tree=False)
                    else:
                        vlogits = model(xv)
                    vl.append(F.cross_entropy(vlogits.view(-1, vocab_size), yv.view(-1)).item())
                val_loss = sum(vl) / len(vl)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

            progress = min(total_training_time / time_budget, 1.0)
            print(f"  step {step:5d} | loss {loss.item():.4f} | val_loss {val_loss:.4f} | {progress:.0%}")
            model.train()

        step += 1
        if step > 2 and total_training_time >= time_budget:
            break

    # Final evaluation (more batches for stable estimate)
    model.eval()
    with torch.no_grad():
        vl = []
        for _ in range(EVAL_BATCHES * 4):
            xv, yv = get_batch(val_data, BATCH_SIZE, MAX_SEQ_LEN)
            if is_h4:
                vlogits = model(xv, use_tree=False)
            else:
                vlogits = model(xv)
            vl.append(F.cross_entropy(vlogits.view(-1, vocab_size), yv.view(-1)).item())
        final_val_loss = sum(vl) / len(vl)

    val_bpb = final_val_loss / math.log(2)
    perplexity = math.exp(final_val_loss)

    # Generate sample
    seed_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    if is_h4:
        gen = model.generate(seed_ids, max_new_tokens=60, temperature=0.8, top_k_sample=10)
    else:
        gen = model.generate(seed_ids, max_new_tokens=60, temperature=0.8, top_k_sample=10)
    gen_text = ''.join([itos.get(i.item(), '?') for i in gen[0]])

    wall_time = time.time() - t_start

    results = {
        'name': name,
        'attention': config['attention'],
        'bitlinear': config['bitlinear'],
        'params': param_info['trainable'],
        'steps': step,
        'val_loss': final_val_loss,
        'best_val_loss': best_val_loss,
        'val_bpb': val_bpb,
        'perplexity': perplexity,
        'wall_time': wall_time,
        'train_time': total_training_time,
        'sample': gen_text[:100],
    }

    print(f"  Final: val_loss={final_val_loss:.4f}, bpb={val_bpb:.4f}, "
          f"ppl={perplexity:.1f}, steps={step}, time={wall_time:.0f}s")

    return results


def print_comparison_table(all_results, dataset_name, time_budget=TIME_BUDGET):
    """Print a formatted comparison table."""
    print(f"\n{'='*80}")
    print(f"COMPARISON RESULTS — Dataset: {dataset_name}")
    print(f"Config: d_model={D_MODEL}, n_layers={N_LAYERS}, n_heads={N_HEADS}, "
          f"seq_len={MAX_SEQ_LEN}, budget={time_budget}s")
    print(f"{'='*80}")

    # Header
    print(f"{'Model':<16} {'Params':>8} {'Steps':>6} {'Val Loss':>9} "
          f"{'BPB':>7} {'PPL':>8} {'Time':>6}")
    print(f"{'-'*16} {'-'*8} {'-'*6} {'-'*9} {'-'*7} {'-'*8} {'-'*6}")

    # Sort by val_loss
    sorted_results = sorted(all_results, key=lambda r: r['val_loss'])

    for r in sorted_results:
        params_str = f"{r['params'] // 1000}K" if r['params'] >= 1000 else str(r['params'])
        print(f"{r['name']:<16} {params_str:>8} {r['steps']:>6} {r['val_loss']:>9.4f} "
              f"{r['val_bpb']:>7.4f} {r['perplexity']:>8.1f} {r['wall_time']:>5.0f}s")

    # Best model
    best = sorted_results[0]
    print(f"\nBest: {best['name']} (val_loss={best['val_loss']:.4f}, ppl={best['perplexity']:.1f})")

    # H4 vs Softmax comparison
    h4_float = next((r for r in all_results if r['attention'] == 'h4' and not r['bitlinear']), None)
    softmax = next((r for r in all_results if r['attention'] == 'softmax'), None)
    if h4_float and softmax:
        delta = softmax['val_loss'] - h4_float['val_loss']
        pct = (delta / softmax['val_loss']) * 100
        if delta > 0:
            print(f"H4 Float vs Softmax: H4 wins by {delta:.4f} nats ({pct:.1f}% better)")
        else:
            print(f"H4 Float vs Softmax: Softmax wins by {-delta:.4f} nats ({-pct:.1f}% better)")

    # Sample text from each model
    print(f"\n{'='*80}")
    print("GENERATED SAMPLES:")
    print(f"{'='*80}")
    for r in sorted_results:
        print(f"\n[{r['name']}]")
        print(f"  {r['sample']}")


def main():
    parser = argparse.ArgumentParser(description='Compare H4 vs baseline attention mechanisms')
    parser.add_argument('--dataset', default='shakespeare',
                        choices=['synthetic', 'shakespeare', 'tinystories'],
                        help='Dataset to use (default: shakespeare)')
    parser.add_argument('--time-budget', type=int, default=TIME_BUDGET,
                        help=f'Training time per model in seconds (default: {TIME_BUDGET})')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Subset of models to run (e.g., "h4 softmax")')
    args = parser.parse_args()

    time_budget = args.time_budget

    print(f"H4 Polytopic Attention — Baseline Comparison")
    print(f"Dataset: {args.dataset}, Time budget: {time_budget}s per model")
    print(f"Expected total time: ~{len(CONFIGS) * time_budget // 60} minutes")

    # Load data
    train_data, val_data, vocab_size, stoi, itos = load_and_prepare(args.dataset)
    print(f"Vocab: {vocab_size}, Train: {len(train_data):,}, Val: {len(val_data):,}")

    # Filter configs if requested
    configs = CONFIGS
    if args.models:
        configs = [c for c in CONFIGS if any(m.lower() in c['name'].lower() for m in args.models)]
        if not configs:
            print(f"No matching models for {args.models}. Available: {[c['name'] for c in CONFIGS]}")
            return

    # Run comparisons
    all_results = []
    for config in configs:
        try:
            results = train_and_evaluate(
                config, train_data, val_data, vocab_size, itos, time_budget
            )
            all_results.append(results)
        except Exception as e:
            print(f"\n  ERROR training {config['name']}: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        print_comparison_table(all_results, args.dataset, time_budget)


if __name__ == '__main__':
    main()
