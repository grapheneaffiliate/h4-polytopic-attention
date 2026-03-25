"""
Train H4 geometric ranker with contrastive learning on SQuAD.

For each batch:
    - Each question is paired with its correct passage (positive)
    - All other passages in the batch are negatives (in-batch negatives)
    - Loss: InfoNCE — correct passage should score highest

Metric: Recall@1 (does the top-ranked passage contain the answer?)

This is a much simpler task than extractive QA:
    - Ranking maps two texts to a scalar (not text to text)
    - 370K ternary params can learn this
    - 5,928 SQuAD pairs provide enough signal
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

from rag.ranking_model import H4Ranker
from rag.prepare_qa import download_squad_dev
from rag.tokenizer import BPETokenizer

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

TIME_BUDGET = int(os.environ.get('RANKER_TIME', 600))  # default 10 min, override with env
D_MODEL = int(os.environ.get('RANKER_DMODEL', 128))
N_HEADS = 8
N_LAYERS = int(os.environ.get('RANKER_LAYERS', 2))
D_VALUE = D_MODEL // N_HEADS
D_FFN = D_MODEL * 4
USE_BITLINEAR = True
LR = float(os.environ.get('RANKER_LR', 3e-3))
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
BATCH_SIZE = int(os.environ.get('RANKER_BATCH', 32))
MAX_Q_LEN = 64
MAX_P_LEN = 192
TEMPERATURE = 0.15
EVAL_INTERVAL = 200


def pad_tokens(ids, max_len):
    """Pad or truncate token list to fixed length."""
    ids = ids[:max_len]
    return ids + [0] * (max_len - len(ids))


def contrastive_loss(q_h4, p_h4, temperature):
    """InfoNCE loss with in-batch negatives."""
    B = q_h4.shape[0]
    # Similarity matrix: (B, B)
    sim = torch.mm(q_h4, p_h4.t()) / temperature

    # Labels: positive is on diagonal
    labels = torch.arange(B, device=sim.device)
    loss = F.cross_entropy(sim, labels)

    # Metrics
    with torch.no_grad():
        preds = sim.argmax(dim=1)
        recall_at_1 = (preds == labels).float().mean().item()

        top5 = sim.topk(min(5, B), dim=1).indices
        recall_at_5 = sum(
            labels[i].item() in top5[i].tolist() for i in range(B)
        ) / B

        ranks = (sim.argsort(dim=1, descending=True) == labels.unsqueeze(1)).nonzero()[:, 1].float() + 1
        mrr = (1.0 / ranks).mean().item()

    return loss, recall_at_1, recall_at_5, mrr


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
    tokenizer = BPETokenizer(max_vocab=4096)
    all_texts = [qa['context'] + ' ' + qa['question'] for qa in squad[:2000]]
    tokenizer.build_vocab(all_texts)
    vocab_size = tokenizer.vocab_size

    # Split train/val
    indices = list(range(len(squad)))
    random.shuffle(indices)
    n_val = 200
    val_indices = set(indices[:n_val])
    train_data = [squad[i] for i in indices if i not in val_indices]
    val_data = [squad[i] for i in indices if i in val_indices]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Create model
    model = H4Ranker(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_value=D_VALUE,
        d_ffn=D_FFN,
        use_bitlinear=USE_BITLINEAR,
        max_seq_len=max(MAX_Q_LEN, MAX_P_LEN),
    )
    n_params = model.count_params()
    print(f"Model: {n_params:,} params ({'ternary' if USE_BITLINEAR else 'float'})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))

    def lr_schedule(step):
        if step < 50:
            return step / 50
        progress = (step - 50) / max(1, 5000 - 50)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Training loop
    model.train()
    step = 0
    total_training_time = 0.0
    best_r1 = 0.0

    print(f"\nTraining for {TIME_BUDGET}s, metric=Recall@1")
    print(f"{'step':>6} {'loss':>8} {'R@1':>8} {'R@5':>8} {'MRR':>8} {'lr':>10}")
    print("-" * 56)

    while True:
        t0 = time.time()

        # Sample batch
        batch = random.sample(train_data, min(BATCH_SIZE, len(train_data)))

        q_ids = torch.tensor(
            [pad_tokens(tokenizer.encode(qa['question']), MAX_Q_LEN) for qa in batch],
            dtype=torch.long,
        )
        p_ids = torch.tensor(
            [pad_tokens(tokenizer.encode(qa['context']), MAX_P_LEN) for qa in batch],
            dtype=torch.long,
        )

        # Encode
        q_h4 = model.encode(q_ids)
        p_h4 = model.encode(p_ids)

        # Loss
        loss, r1, r5, mrr = contrastive_loss(q_h4, p_h4, TEMPERATURE)

        optimizer.zero_grad()
        loss.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        dt = time.time() - t0
        if step > 2:
            total_training_time += dt

        if step % EVAL_INTERVAL == 0:
            # Quick val eval
            model.eval()
            val_r1s = []
            with torch.no_grad():
                for vi in range(0, min(len(val_data), 100), BATCH_SIZE):
                    vbatch = val_data[vi:vi + BATCH_SIZE]
                    if len(vbatch) < 2:
                        continue
                    vq = torch.tensor(
                        [pad_tokens(tokenizer.encode(qa['question']), MAX_Q_LEN) for qa in vbatch],
                        dtype=torch.long,
                    )
                    vp = torch.tensor(
                        [pad_tokens(tokenizer.encode(qa['context']), MAX_P_LEN) for qa in vbatch],
                        dtype=torch.long,
                    )
                    vq_h4 = model.encode(vq)
                    vp_h4 = model.encode(vp)
                    _, vr1, _, _ = contrastive_loss(vq_h4, vp_h4, TEMPERATURE)
                    val_r1s.append(vr1)

            val_r1 = sum(val_r1s) / len(val_r1s) if val_r1s else 0
            if val_r1 > best_r1:
                best_r1 = val_r1

            current_lr = scheduler.get_last_lr()[0]
            print(f"{step:6d} {loss.item():8.4f} {val_r1:8.3f} {r5:8.3f} {mrr:8.3f} {current_lr:10.6f}")
            model.train()

        step += 1
        if step > 2 and total_training_time >= TIME_BUDGET:
            break

    # Final evaluation
    model.eval()
    print("\n" + "=" * 60)
    print("FINAL RANKING EVALUATION:")

    all_r1 = []
    all_r5 = []
    all_mrr = []
    with torch.no_grad():
        for vi in range(0, min(len(val_data), 200), BATCH_SIZE):
            vbatch = val_data[vi:vi + BATCH_SIZE]
            if len(vbatch) < 2:
                continue
            vq = torch.tensor(
                [pad_tokens(tokenizer.encode(qa['question']), MAX_Q_LEN) for qa in vbatch],
                dtype=torch.long,
            )
            vp = torch.tensor(
                [pad_tokens(tokenizer.encode(qa['context']), MAX_P_LEN) for qa in vbatch],
                dtype=torch.long,
            )
            vq_h4 = model.encode(vq)
            vp_h4 = model.encode(vp)
            _, vr1, vr5, vmrr = contrastive_loss(vq_h4, vp_h4, TEMPERATURE)
            all_r1.append(vr1)
            all_r5.append(vr5)
            all_mrr.append(vmrr)

    final_r1 = sum(all_r1) / len(all_r1) if all_r1 else 0
    final_r5 = sum(all_r5) / len(all_r5) if all_r5 else 0
    final_mrr = sum(all_mrr) / len(all_mrr) if all_mrr else 0

    # Show some examples
    print(f"\nSample rankings (batch of {min(BATCH_SIZE, 8)}):")
    sample_batch = val_data[:min(BATCH_SIZE, 8)]
    sq = torch.tensor([pad_tokens(tokenizer.encode(qa['question']), MAX_Q_LEN) for qa in sample_batch], dtype=torch.long)
    sp = torch.tensor([pad_tokens(tokenizer.encode(qa['context']), MAX_P_LEN) for qa in sample_batch], dtype=torch.long)
    with torch.no_grad():
        sq_h4 = model.encode(sq)
        sp_h4 = model.encode(sp)
        sim = torch.mm(sq_h4, sp_h4.t())

    for i in range(min(3, len(sample_batch))):
        scores = sim[i].tolist()
        ranked = sorted(range(len(scores)), key=lambda j: -scores[j])
        correct = i
        rank_of_correct = ranked.index(correct) + 1
        print(f"  Q: {sample_batch[i]['question'][:60]}")
        print(f"  Correct passage rank: {rank_of_correct}/{len(scores)}")
        print(f"  Top passage: {sample_batch[ranked[0]]['context'][:60]}...")
        print()

    print("=" * 60)
    print("\n---")
    print(f"val_recall_at_1:    {final_r1:.4f}")
    print(f"val_recall_at_5:    {final_r5:.4f}")
    print(f"val_mrr:            {final_mrr:.4f}")
    print(f"best_recall_at_1:   {best_r1:.4f}")
    print(f"training_seconds:   {total_training_time:.1f}")
    print(f"total_seconds:      {time.time() - t_start:.1f}")
    print(f"num_steps:          {step}")
    print(f"num_params:         {n_params}")
    print(f"ternary:            {'yes' if USE_BITLINEAR else 'no'}")
    print(f"batch_size:         {BATCH_SIZE}")
    print(f"temperature:        {TEMPERATURE}")


if __name__ == '__main__':
    main()
