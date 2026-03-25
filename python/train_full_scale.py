"""
Full-scale H4 language model training on TinyStories.

Not an autoresearch experiment. A single long training run at real scale.
Saves checkpoints every 30 minutes, evaluates perplexity, generates samples.

Usage:
    python train_full_scale.py                    # 8 hours, d_model=1024
    python train_full_scale.py --time 1800        # 30-minute simulation
    python train_full_scale.py --d_model 512      # smaller model, faster steps
"""

import os
import math
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h4_language_model import H4LanguageModel
from rag.tokenizer import BPETokenizer


def load_tinystories(tokenizer, max_tokens=None):
    """Load TinyStories train/val with BPE tokenization."""
    train_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tinystories_train.txt')
    val_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tinystories_valid.txt')

    # Fall back to Shakespeare if TinyStories not available
    if not os.path.exists(train_path):
        train_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'shakespeare.txt')
        val_path = train_path
        print(f"TinyStories not found, using Shakespeare")

    print(f"Loading training data from {train_path}...")
    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_text = f.read()

    print(f"Loading validation data from {val_path}...")
    with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
        val_text = f.read()

    # Build vocab from training data
    print("Building BPE vocabulary...")
    # Sample from training text for vocab building
    sample_size = min(len(train_text), 2_000_000)
    tokenizer.build_vocab([train_text[:sample_size]])

    # Tokenize
    print("Tokenizing training data...")
    train_ids = tokenizer.encode(train_text[:min(len(train_text), 10_000_000)])
    if max_tokens:
        train_ids = train_ids[:max_tokens]

    print("Tokenizing validation data...")
    val_ids = tokenizer.encode(val_text[:min(len(val_text), 1_000_000)])

    train_data = torch.tensor(train_ids, dtype=torch.long)
    val_data = torch.tensor(val_ids, dtype=torch.long)

    print(f"Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")
    return train_data, val_data


def get_batch(data, batch_size, seq_len):
    """Sample a random batch."""
    max_start = len(data) - seq_len - 1
    if max_start <= 0:
        max_start = 1
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(model, val_data, vocab_size, batch_size, seq_len, n_batches=20):
    """Evaluate perplexity on validation data."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for _ in range(n_batches):
        x, y = get_batch(val_data, batch_size, seq_len)
        logits = model(x, use_tree=False)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), reduction='sum')
        total_loss += loss.item()
        total_tokens += y.numel()
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20))  # cap to avoid overflow
    return avg_loss, perplexity


@torch.no_grad()
def generate_samples(model, tokenizer, n_samples=3, max_tokens=100, temperature=0.8):
    """Generate text samples from the model."""
    model.eval()
    samples = []
    prompts = ["Once upon a time", "The little girl", "One day, a"]
    for i in range(n_samples):
        prompt = prompts[i % len(prompts)]
        ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], dtype=torch.long)
        output = model.generate(input_ids, max_new_tokens=max_tokens,
                                temperature=temperature, top_k_sample=40)
        text = tokenizer.decode(output[0].tolist())
        samples.append(text)
    return samples


def main():
    parser = argparse.ArgumentParser(description='Full-scale H4 training')
    parser.add_argument('--time', type=int, default=28800, help='Training time in seconds (default: 8h)')
    parser.add_argument('--d_model', type=int, default=1024, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--seq_len', type=int, default=512, help='Sequence length')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--grad_accum', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--checkpoint_interval', type=int, default=1800, help='Checkpoint every N seconds')
    parser.add_argument('--vocab_size', type=int, default=8192, help='BPE vocabulary size')
    parser.add_argument('--float', action='store_true', help='Use float instead of ternary')
    args = parser.parse_args()

    t_start = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    use_bitlinear = not args.float
    d_ffn = args.d_model * 4
    d_value = args.d_model // args.n_heads

    print("=" * 70)
    print("  H4 POLYTOPIC ATTENTION — FULL SCALE TRAINING")
    print("=" * 70)
    print(f"  Model: d_model={args.d_model}, {args.n_layers} layers, {args.n_heads} heads")
    print(f"  Ternary: {'yes' if use_bitlinear else 'no'}")
    print(f"  Seq len: {args.seq_len}, Batch: {args.batch_size} x {args.grad_accum} accum")
    print(f"  LR: {args.lr}, Time budget: {args.time}s ({args.time/3600:.1f}h)")
    print(f"  Checkpoint every: {args.checkpoint_interval}s ({args.checkpoint_interval/60:.0f}min)")
    print()

    # Tokenizer
    tokenizer = BPETokenizer(max_vocab=args.vocab_size)

    # Data
    train_data, val_data = load_tinystories(tokenizer)

    # Model
    model = H4LanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_value=d_value,
        d_ffn=d_ffn,
        max_seq_len=args.seq_len,
        dropout=0.0,
        use_bitlinear=use_bitlinear,
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} total, {n_trainable:,} trainable")

    # Estimate memory
    param_mb = n_params * 4 / 1024 / 1024  # float32 shadow weights
    print(f"  Est memory: {param_mb:.0f} MB (float32 shadow)")

    # Test step time
    print("  Testing step time...")
    x_test = torch.randint(0, tokenizer.vocab_size, (args.batch_size, args.seq_len))
    t0 = time.perf_counter()
    logits = model(x_test, use_tree=False)
    loss = logits.sum() * 0  # dummy
    loss.backward()
    step_ms = (time.perf_counter() - t0) * 1000
    model.zero_grad()

    est_steps_per_hour = 3600 / (step_ms / 1000)
    est_total_steps = args.time / (step_ms / 1000)
    print(f"  Step time: {step_ms:.0f}ms")
    print(f"  Est steps/hour: {est_steps_per_hour:.0f}")
    print(f"  Est total steps: {est_total_steps:.0f}")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    total_steps_est = int(est_total_steps)
    warmup_steps = min(500, total_steps_est // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(1, total_steps_est - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Training
    os.makedirs('checkpoints', exist_ok=True)
    model.train()
    step = 0
    accum_step = 0
    total_training_time = 0.0
    last_checkpoint_time = 0.0
    total_tokens = 0
    best_val_loss = float('inf')
    log_interval = max(10, int(100 / args.grad_accum))

    print(f"{'step':>7} {'loss':>8} {'val_loss':>8} {'ppl':>8} {'lr':>10} {'tok/s':>8} {'elapsed':>10}")
    print("-" * 72)

    optimizer.zero_grad()

    while True:
        t0 = time.time()

        # Get batch
        x, y = get_batch(train_data, args.batch_size, args.seq_len)

        # Forward
        logits = model(x, use_tree=False)
        loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1))
        loss = loss / args.grad_accum  # scale for accumulation

        # Backward
        loss.backward()
        accum_step += 1

        if accum_step >= args.grad_accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accum_step = 0
            step += 1

        dt = time.time() - t0
        if step > 2:
            total_training_time += dt
        total_tokens += args.batch_size * args.seq_len

        # Log
        if step > 0 and step % log_interval == 0 and accum_step == 0:
            elapsed = time.time() - t_start
            toks_per_sec = total_tokens / max(elapsed, 1)
            current_lr = scheduler.get_last_lr()[0]

            # Quick val eval every 5x log interval
            if step % (log_interval * 5) == 0:
                val_loss, ppl = evaluate(model, val_data, tokenizer.vocab_size,
                                         args.batch_size, args.seq_len, n_batches=10)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                print(f"{step:7d} {loss.item()*args.grad_accum:8.4f} {val_loss:8.4f} "
                      f"{ppl:8.1f} {current_lr:10.6f} {toks_per_sec:8.0f} {elapsed:9.0f}s")
                model.train()
            else:
                print(f"{step:7d} {loss.item()*args.grad_accum:8.4f} {'':>8} "
                      f"{'':>8} {current_lr:10.6f} {toks_per_sec:8.0f} {elapsed:9.0f}s")

        # Checkpoint
        elapsed = time.time() - t_start
        if elapsed - last_checkpoint_time >= args.checkpoint_interval and step > 0:
            last_checkpoint_time = elapsed
            ckpt_name = f"h4_fullscale_step{step}.pt"
            ckpt_path = os.path.join('checkpoints', ckpt_name)

            # Evaluate
            val_loss, ppl = evaluate(model, val_data, tokenizer.vocab_size,
                                     args.batch_size, args.seq_len, n_batches=30)

            # Generate samples
            print(f"\n{'='*70}")
            print(f"  CHECKPOINT at step {step} ({elapsed/60:.0f} min)")
            print(f"  Val loss: {val_loss:.4f}, Perplexity: {ppl:.1f}")
            print(f"  Total tokens: {total_tokens:,}")
            print(f"\n  Generated samples:")
            samples = generate_samples(model, tokenizer, n_samples=3, max_tokens=80)
            for i, s in enumerate(samples):
                print(f"  [{i+1}] {s[:200]}")
            print(f"{'='*70}\n")

            # Save checkpoint
            torch.save({
                'model_state': model.state_dict(),
                'step': step,
                'val_loss': val_loss,
                'perplexity': ppl,
                'total_tokens': total_tokens,
                'config': {
                    'd_model': args.d_model, 'n_layers': args.n_layers,
                    'n_heads': args.n_heads, 'vocab_size': tokenizer.vocab_size,
                    'use_bitlinear': use_bitlinear,
                },
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")
            model.train()

        # Time check
        if elapsed >= args.time:
            break

    # Final evaluation
    model.eval()
    val_loss, ppl = evaluate(model, val_data, tokenizer.vocab_size,
                             args.batch_size, args.seq_len, n_batches=50)

    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Steps: {step}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Time: {(time.time()-t_start)/3600:.2f} hours")
    print(f"  Final val loss: {val_loss:.4f}")
    print(f"  Final perplexity: {ppl:.1f}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Parameters: {n_params:,} ({'ternary' if use_bitlinear else 'float'})")

    print(f"\n  Final generated samples:")
    samples = generate_samples(model, tokenizer, n_samples=5, max_tokens=120)
    for i, s in enumerate(samples):
        print(f"  [{i+1}] {s[:250]}")

    # Save final checkpoint
    final_path = os.path.join('checkpoints', 'h4_fullscale_final.pt')
    torch.save({
        'model_state': model.state_dict(),
        'step': step,
        'val_loss': val_loss,
        'perplexity': ppl,
        'total_tokens': total_tokens,
        'config': {
            'd_model': args.d_model, 'n_layers': args.n_layers,
            'n_heads': args.n_heads, 'vocab_size': tokenizer.vocab_size,
            'use_bitlinear': use_bitlinear,
        },
    }, final_path)
    print(f"\n  Final checkpoint: {final_path}")

    print(f"\n---")
    print(f"val_loss:         {val_loss:.4f}")
    print(f"perplexity:       {ppl:.1f}")
    print(f"best_val_loss:    {best_val_loss:.4f}")
    print(f"total_steps:      {step}")
    print(f"total_tokens:     {total_tokens}")
    print(f"training_hours:   {(time.time()-t_start)/3600:.2f}")
    print(f"params:           {n_params}")
    print(f"ternary:          {'yes' if use_bitlinear else 'no'}")
    print(f"d_model:          {args.d_model}")
    print(f"n_layers:         {args.n_layers}")


if __name__ == '__main__':
    main()
