"""
Train H4 cross-encoder reranker on SQuAD.

Uses the PPL 10.0 TinyStories checkpoint as backbone.
Fine-tunes on binary classification: does this passage answer this question?

For each SQuAD example:
  - Positive: [question SEP correct_passage] -> label 1
  - Negative: [question SEP wrong_passage] -> label 0

The H4 attention heads directly attend from question tokens to passage tokens
within the same sequence — this is why cross-encoders beat bi-encoders.

Pipeline integration:
  1. Bi-encoder retrieves top-5 (R@5=100%, 20ms)
  2. Cross-encoder reranks 5 candidates (5 forward passes, ~50ms)
  3. Return highest-scoring → R@1 should reach 80-90%+
"""

import os
import math
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag.cross_encoder import H4CrossEncoder
from rag.prepare_qa import download_squad_dev
from rag.tokenizer import BPETokenizer

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

TIME_BUDGET = int(os.environ.get('CE_TIME', 3600))  # 1 hour default
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 8
USE_BITLINEAR = True
LR = 5e-4  # lower LR for fine-tuning (backbone is pre-trained)
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
BATCH_SIZE = 8  # pairs per batch (each has 1 positive + 1 negative)
MAX_SEQ_LEN = 192  # question + passage combined
EVAL_INTERVAL = 100
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'h4_fullscale_final.pt')


def pad_tokens(ids, max_len):
    ids = ids[:max_len]
    return ids + [0] * (max_len - len(ids))


def make_pair(tokenizer, question, passage, max_len):
    """Encode [question SEP passage] as a single sequence."""
    q_ids = tokenizer.encode(question)
    p_ids = tokenizer.encode(passage)
    # Budget: half for question, half for passage (with SEP)
    max_q = max_len // 3
    max_p = max_len - max_q - 1
    q_ids = q_ids[:max_q]
    p_ids = p_ids[:max_p]
    combined = q_ids + [2] + p_ids  # 2 = SEP
    return pad_tokens(combined, max_len)


def main():
    t_start = time.time()
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Load SQuAD
    squad = download_squad_dev()
    if len(squad) < 100:
        print("SQuAD not available. Run: python rag/prepare_qa.py")
        return
    print(f"SQuAD: {len(squad)} QA pairs")

    # Build BPE tokenizer
    tokenizer = BPETokenizer(max_vocab=8192)
    all_texts = [qa['context'] + ' ' + qa['question'] for qa in squad[:2000]]
    tokenizer.build_vocab(all_texts)

    # Split
    indices = list(range(len(squad)))
    random.shuffle(indices)
    n_val = 200
    train_data = [squad[i] for i in indices[n_val:]]
    val_data = [squad[i] for i in indices[:n_val]]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Create cross-encoder
    model = H4CrossEncoder(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        use_bitlinear=USE_BITLINEAR,
        max_seq_len=MAX_SEQ_LEN,
    )
    print(f"Model: {model.count_params():,} params")

    # Load pre-trained backbone
    if os.path.exists(CHECKPOINT_PATH):
        config = model.load_lm_backbone(CHECKPOINT_PATH)
        print(f"Loaded backbone from {CHECKPOINT_PATH}")
    else:
        print(f"No checkpoint at {CHECKPOINT_PATH}, training from scratch")

    # Freeze backbone initially, only train score head
    # Then unfreeze after warmup for fine-tuning
    for name, param in model.lm.named_parameters():
        param.requires_grad = False
    trainable_head = sum(p.numel() for p in model.score_head.parameters())
    print(f"Phase 1: training score head only ({trainable_head:,} params)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR * 10,  # higher LR for head-only phase
        weight_decay=WEIGHT_DECAY,
    )

    # Training loop
    model.train()
    step = 0
    total_training_time = 0.0
    best_acc = 0.0
    unfrozen = False
    UNFREEZE_STEP = 200

    print(f"\nTraining for {TIME_BUDGET}s")
    print(f"{'step':>6} {'loss':>8} {'acc':>8} {'val_acc':>8} {'phase':>10}")
    print("-" * 48)

    while True:
        t0 = time.time()

        # Unfreeze backbone after warmup
        if step == UNFREEZE_STEP and not unfrozen:
            for param in model.lm.parameters():
                param.requires_grad = True
            unfrozen = True
            total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n  Phase 2: unfreezing backbone ({total_trainable:,} trainable params)")
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))

        # Sample batch: positive and negative pairs
        batch_qa = random.sample(train_data, min(BATCH_SIZE, len(train_data)))
        input_ids = []
        labels = []

        for qa in batch_qa:
            # Positive: question + correct passage
            pos = make_pair(tokenizer, qa['question'], qa['context'], MAX_SEQ_LEN)
            input_ids.append(pos)
            labels.append(1.0)

            # Negative: question + random wrong passage
            neg_qa = random.choice(train_data)
            while neg_qa['context'] == qa['context']:
                neg_qa = random.choice(train_data)
            neg = make_pair(tokenizer, qa['question'], neg_qa['context'], MAX_SEQ_LEN)
            input_ids.append(neg)
            labels.append(0.0)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.float32)

        # Forward
        scores = model(input_ids)
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        # Accuracy
        with torch.no_grad():
            preds = (scores > 0).float()
            acc = (preds == labels).float().mean().item()

        optimizer.zero_grad()
        loss.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        dt = time.time() - t0
        if step > 2:
            total_training_time += dt

        # Eval
        if step % EVAL_INTERVAL == 0:
            model.eval()
            val_correct = 0
            val_total = 0
            val_r1 = 0
            val_r1_total = 0

            with torch.no_grad():
                # Binary accuracy
                for vi in range(0, min(len(val_data), 100), BATCH_SIZE):
                    vbatch = val_data[vi:vi + BATCH_SIZE]
                    v_ids = []
                    v_labels = []
                    for qa in vbatch:
                        pos = make_pair(tokenizer, qa['question'], qa['context'], MAX_SEQ_LEN)
                        v_ids.append(pos)
                        v_labels.append(1.0)
                        neg_qa = random.choice(val_data)
                        neg = make_pair(tokenizer, qa['question'], neg_qa['context'], MAX_SEQ_LEN)
                        v_ids.append(neg)
                        v_labels.append(0.0)
                    v_ids = torch.tensor(v_ids, dtype=torch.long)
                    v_labels = torch.tensor(v_labels)
                    v_scores = model(v_ids)
                    v_preds = (v_scores > 0).float()
                    val_correct += (v_preds == v_labels).sum().item()
                    val_total += len(v_labels)

                # Reranking accuracy (R@1 on top-5 candidates)
                for qa in val_data[:50]:
                    # Simulate: 1 correct + 4 wrong passages
                    candidates = [qa['context']]
                    neg_pool = [q for q in val_data if q['context'] != qa['context']]
                    for neg in random.sample(neg_pool, min(4, len(neg_pool))):
                        candidates.append(neg['context'])

                    c_ids = []
                    for passage in candidates:
                        c_ids.append(make_pair(tokenizer, qa['question'], passage, MAX_SEQ_LEN))
                    c_ids = torch.tensor(c_ids, dtype=torch.long)
                    c_scores = model(c_ids)
                    top_idx = c_scores.argmax().item()
                    if top_idx == 0:  # correct passage was ranked first
                        val_r1 += 1
                    val_r1_total += 1

            val_acc = val_correct / max(val_total, 1)
            rerank_r1 = val_r1 / max(val_r1_total, 1)

            if val_acc > best_acc:
                best_acc = val_acc

            phase = "head-only" if not unfrozen else "full"
            print(f"{step:6d} {loss.item():8.4f} {acc:8.3f} {val_acc:8.3f} {phase:>10}"
                  f"  rerank_R@1={rerank_r1:.3f}")
            model.train()

        step += 1
        elapsed = time.time() - t_start
        if step > 2 and total_training_time >= TIME_BUDGET:
            break

    # Final evaluation
    model.eval()
    print("\n" + "=" * 60)
    print("FINAL CROSS-ENCODER EVALUATION:")

    final_r1 = 0
    final_total = 0
    with torch.no_grad():
        for qa in val_data[:100]:
            candidates = [qa['context']]
            neg_pool = [q for q in val_data if q['context'] != qa['context']]
            for neg in random.sample(neg_pool, min(4, len(neg_pool))):
                candidates.append(neg['context'])

            c_ids = []
            for passage in candidates:
                c_ids.append(make_pair(tokenizer, qa['question'], passage, MAX_SEQ_LEN))
            c_ids = torch.tensor(c_ids, dtype=torch.long)
            c_scores = model(c_ids)
            if c_scores.argmax().item() == 0:
                final_r1 += 1
            final_total += 1

    rerank_r1 = final_r1 / max(final_total, 1)

    print(f"  Rerank R@1 (top-5): {rerank_r1:.1%} ({final_r1}/{final_total})")
    print(f"  Best binary acc: {best_acc:.1%}")
    print("=" * 60)

    print("\n---")
    print(f"rerank_r1:        {rerank_r1:.4f}")
    print(f"best_binary_acc:  {best_acc:.4f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {time.time() - t_start:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params:       {model.count_params()}")
    print(f"ternary:          {'yes' if USE_BITLINEAR else 'no'}")

    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    ckpt_path = os.path.join('checkpoints', 'h4_cross_encoder.pt')
    torch.save({
        'model_state': model.state_dict(),
        'rerank_r1': rerank_r1,
        'step': step,
        'config': {
            'd_model': D_MODEL, 'n_layers': N_LAYERS, 'n_heads': N_HEADS,
            'vocab_size': tokenizer.vocab_size, 'use_bitlinear': USE_BITLINEAR,
        },
    }, ckpt_path)
    print(f"Saved: {ckpt_path}")


if __name__ == '__main__':
    main()
