"""
Train H4 attention model for extractive question-answering.

Uses the autoresearch pattern: modify -> run (2 min) -> measure -> keep/discard.
Metric: F1 score on validation QA pairs (not bpb).

The model learns to generate answer text given [context | question |] as input.
This is extractive QA — the answer is a span from the context.

Architecture: same H4LanguageModel from Phase 5/6, trained on QA-formatted text.
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import re
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from h4_language_model import H4LanguageModel
from rag.prepare_qa import generate_sample_qa, prepare_training_data, format_qa_for_training, download_squad_dev
from rag.tokenizer import BPETokenizer

# ---------------------------------------------------------------------------
# Hyperparameters (AGENT MAY MODIFY THESE)
# ---------------------------------------------------------------------------

TIME_BUDGET = 600  # 10 minutes for QA fine-tuning

# Model (must match pre-trained checkpoint)
D_MODEL = 64
N_HEADS = 8
N_LAYERS = 2
D_VALUE = 16
D_FFN = 256
DROPOUT = 0.0
USE_BITLINEAR = True

# Pre-trained checkpoint (set to None to train from scratch)
PRETRAINED_CKPT = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'lm_pretrained.pt')

# Optimizer (lower LR for fine-tuning from pre-trained)
LR = 3e-3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 30
GRAD_CLIP = 1.0

# Training
BATCH_SIZE = 2
MAX_SEQ_LEN = 512
EVAL_INTERVAL = 500

# ---------------------------------------------------------------------------
# QA-specific metrics
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lower case, strip whitespace and punctuation."""
    s = s.lower().strip()
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s


def compute_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and gold answer."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())

    if n_common == 0:
        return 0.0

    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_exact_match(prediction: str, gold: str) -> float:
    """Exact match after normalization."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    # Load QA data — use SQuAD if available, fall back to sample
    squad_qa = download_squad_dev()
    if len(squad_qa) > 100:
        print(f"Using SQuAD 2.0: {len(squad_qa)} QA pairs")
        all_qa = squad_qa
    else:
        print("SQuAD not available, using sample QA pairs")
        all_qa = generate_sample_qa()

    # Build BPE tokenizer from training data
    tokenizer = BPETokenizer(max_vocab=4096)
    all_texts = [qa['context'] + ' ' + qa['question'] + ' ' + qa['answer'] for qa in all_qa[:2000]]
    tokenizer.build_vocab(all_texts)
    vocab_size = tokenizer.vocab_size

    # Prepare training sequences using BPE
    # Format: [context SEP question SEP answer]
    # With BPE, seq_len=512 covers ~2500 chars — virtually all SQuAD contexts
    all_seqs = []
    for qa in all_qa:
        input_ids, answer_ids = tokenizer.encode_qa(qa['context'], qa['question'], qa['answer'])
        full_ids = input_ids + answer_ids
        if len(full_ids) <= MAX_SEQ_LEN and len(full_ids) > 5:
            all_seqs.append((full_ids, len(input_ids), qa))

    print(f"BPE sequences that fit seq_len={MAX_SEQ_LEN}: {len(all_seqs)} "
          f"(from {len(all_qa)} total QA pairs)")

    # Split into train/val
    rng = np.random.RandomState(42)
    indices = list(range(len(all_seqs)))
    rng.shuffle(indices)
    n_val = min(30, max(20, int(len(all_seqs) * 0.05)))
    val_indices = set(indices[:n_val])

    train_seqs = []
    val_pairs = []
    for i, (full_ids, input_len, qa) in enumerate(all_seqs):
        if i in val_indices:
            val_pairs.append((full_ids[:input_len], qa['answer'], qa))
        else:
            train_seqs.append(torch.tensor(full_ids, dtype=torch.long))

    rng.shuffle(train_seqs)
    avg_len = sum(len(s) for s in train_seqs) / len(train_seqs) if train_seqs else 0
    print(f"Train: {len(train_seqs)} seqs (avg {avg_len:.0f} BPE tokens), Val: {len(val_pairs)} pairs")

    # Create model
    model = H4LanguageModel(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_value=D_VALUE,
        d_ffn=D_FFN,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
        use_bitlinear=USE_BITLINEAR,
    )
    params = model.count_params()
    print(f"Model: {params['trainable']:,} trainable params")

    # Note: pre-trained checkpoint skipped because vocab size changed with BPE.
    # The model trains from scratch but BPE makes this much more efficient.

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))

    def lr_schedule(step):
        if step < WARMUP_STEPS:
            return step / max(WARMUP_STEPS, 1)
        progress = (step - WARMUP_STEPS) / max(1, 5000 - WARMUP_STEPS)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Training loop
    step = 0
    total_training_time = 0.0
    best_f1 = 0.0
    model.train()

    print(f"\nTraining for {TIME_BUDGET}s, metric=F1")
    print(f"{'step':>6} {'loss':>8} {'val_f1':>8} {'val_em':>8} {'lr':>10}")
    print("-" * 48)

    while True:
        t0 = time.time()

        # Sample a training sequence
        seq = train_seqs[step % len(train_seqs)]
        x = seq[:-1].unsqueeze(0)  # (1, T-1)
        y = seq[1:].unsqueeze(0)   # (1, T-1)

        # Pad/truncate to consistent length
        if x.shape[1] > MAX_SEQ_LEN:
            x = x[:, :MAX_SEQ_LEN]
            y = y[:, :MAX_SEQ_LEN]

        logits = model(x, use_tree=False)
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

        # Evaluate
        if step % EVAL_INTERVAL == 0:
            model.eval()
            f1_scores = []
            em_scores = []

            with torch.no_grad():
                for inp_ids, gold_answer, qa in val_pairs:
                    # Generate answer
                    inp_tensor = torch.tensor([inp_ids], dtype=torch.long)
                    max_answer_len = min(len(tokenizer.encode(gold_answer)) + 10, 50)

                    generated = model.generate(
                        inp_tensor,
                        max_new_tokens=max_answer_len,
                        temperature=0.5,
                        top_k_sample=10,
                    )
                    gen_ids = generated[0, len(inp_ids):]
                    pred_answer = tokenizer.decode(gen_ids.tolist())

                    f1 = compute_f1(pred_answer, gold_answer)
                    em = compute_exact_match(pred_answer, gold_answer)
                    f1_scores.append(f1)
                    em_scores.append(em)

            avg_f1 = sum(f1_scores) / len(f1_scores)
            avg_em = sum(em_scores) / len(em_scores)

            if avg_f1 > best_f1:
                best_f1 = avg_f1

            current_lr = scheduler.get_last_lr()[0]
            print(f"{step:6d} {loss.item():8.4f} {avg_f1:8.3f} {avg_em:8.3f} {current_lr:10.6f}")
            model.train()

        step += 1
        if step > 2 and total_training_time >= TIME_BUDGET:
            break

    # Final evaluation with sample outputs
    model.eval()
    print("\n" + "=" * 60)
    print("SAMPLE QA RESULTS:")
    final_f1_scores = []
    final_em_scores = []

    with torch.no_grad():
        for inp_ids, gold_answer, qa in val_pairs:
            inp_tensor = torch.tensor([inp_ids], dtype=torch.long)
            max_answer_len = min(len(gold_answer) + 20, 100)
            generated = model.generate(
                inp_tensor, max_new_tokens=max_answer_len,
                temperature=0.3, top_k_sample=5,
            )
            gen_ids = generated[0, len(inp_ids):]
            pred_answer = tokenizer.decode(gen_ids.tolist())

            f1 = compute_f1(pred_answer, gold_answer)
            em = compute_exact_match(pred_answer, gold_answer)
            final_f1_scores.append(f1)
            final_em_scores.append(em)

            print(f"  Q: {qa['question']}")
            print(f"  Gold: {gold_answer}")
            print(f"  Pred: {pred_answer[:80]}")
            print(f"  F1={f1:.3f} EM={em:.1f}")
            print()

    avg_f1 = sum(final_f1_scores) / len(final_f1_scores) if final_f1_scores else 0
    avg_em = sum(final_em_scores) / len(final_em_scores) if final_em_scores else 0

    print("=" * 60)
    print("\n---")
    print(f"val_f1:             {avg_f1:.4f}")
    print(f"val_em:             {avg_em:.4f}")
    print(f"best_f1:            {best_f1:.4f}")
    print(f"training_seconds:   {total_training_time:.1f}")
    print(f"total_seconds:      {time.time() - t_start:.1f}")
    print(f"num_steps:          {step}")
    print(f"num_params:         {params['trainable']}")
    print(f"ternary:            {'yes' if USE_BITLINEAR else 'no'}")


if __name__ == '__main__':
    main()
