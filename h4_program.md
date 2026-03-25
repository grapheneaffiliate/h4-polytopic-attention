# H4 Polytopic Attention — Autonomous Research Protocol

You are an autonomous research agent optimizing a **hybrid transformer** that combines frozen H4 geometric attention with trainable adapter layers. Your goal is to minimize `val_bpb` (bits per byte on held-out text) within a fixed compute budget.

## The Architecture

The system has two parts:

1. **Frozen geometric backbone** (DO NOT MODIFY):
   - 600-cell vertices (120 × 4) — the H4 polytope
   - H4 simple roots (4 × 4) — Coxeter reflection hyperplanes
   - E8→H4 projection matrix (4 × 8) — golden ratio eigenvalues
   - ChamberTree structure — O(log t) spatial partitioning of S³
   - E8LatticeIndex — Voronoi cell memory backend

2. **Trainable adapters** (YOU MODIFY THESE):
   - `W_q_proj`, `W_k_proj`, `W_v_proj` — input projections to H4/value space
   - `W_nudge` — per-head 4×4 rotation of query direction
   - `chamber_bonus` — per-head 16-dim learnable attention bias per chamber
   - `W_out` — output projection
   - FFN layers (fully trainable)
   - Token/positional embeddings
   - LM head

## What You Can Change

In `python/train_cpu.py`, you may modify:

- **Hyperparameters:** learning rate, batch size, sequence length, warmup, grad clip, weight decay
- **Architecture of trainable layers:** d_model, n_heads, n_layers, d_value, d_ffn, top_k, dropout
- **Optimizer setup:** scheduler, betas, epsilon
- **Training loop:** loss weighting, gradient accumulation, evaluation strategy
- **Adapter architecture:** number of nudge parameters, chamber bonus structure, FFN design

## What You CANNOT Change

- `python/h4_polytopic_attention.py` — frozen geometry
- `python/utils/phi_positional.py` — golden-angle encoding (you can change how it's used, not the encoding itself)
- `python/utils/chamber_index.py` — chamber lookup bridge
- The H4 vertices, simple roots, E8 projection, ChamberTree structure
- The fundamental constraint that Q and K live on S³ (unit 4-sphere)

## The Loop

```
while forever:
    1. Read current train_cpu.py and results.tsv
    2. Form a hypothesis about what change will improve val_bpb
    3. Modify train_cpu.py (the ONLY file you modify)
    4. Run: cd python && python train_cpu.py
    5. Parse the "---" summary block from stdout
    6. If val_bpb improved or the experiment is informative:
         - git add python/train_cpu.py && git commit
         - Append to results.tsv: commit<TAB>val_bpb<TAB>val_loss<TAB>chamber_entropy<TAB>status<TAB>description
         - status = "keep"
    7. If val_bpb did not improve:
         - git checkout python/train_cpu.py  (discard changes)
         - Append to results.tsv with status = "discard"
    8. If the run crashed:
         - Fix the crash, do NOT count it as an experiment
         - Append to results.tsv with status = "crash"
    9. Think about what you learned. Update your mental model of:
         - Which hyperparameters matter most
         - Whether the geometry is being utilized (check chamber_entropy)
         - Whether W_nudge is learning meaningful directions (check geo_alignment)
         - What the loss landscape looks like
   10. Repeat
```

## Time Budget

- **2 minutes per experiment** on CPU (TIME_BUDGET = 120 in train_cpu.py)
- This allows ~24 experiments in an overnight run
- If an experiment takes longer than 3 minutes, it has a bug — fix it

## Metrics

Primary: `val_bpb` — lower is better. Bits per byte on held-out character-level text.

Diagnostic (track but don't optimize directly):
- `chamber_entropy` — Shannon entropy of chamber utilization. High = using full geometry. Low = collapsed to few chambers.
- `avg_nudge_rank` — effective rank of W_nudge deviation from identity. High = rank-1 (good, focused direction). Low = diffuse.
- `avg_geo_alignment` — max dot product of W_nudge dominant direction with 600-cell vertices. >0.9 = strongly aligning with geometry.
- `num_steps` — training throughput indicator.

## results.tsv Format

```
commit	val_bpb	val_loss	chamber_entropy	status	description
a1b2c3d	2.345678	1.625000	2.1234	keep	baseline: d_model=64, 2 layers, lr=3e-4
e4f5g6h	2.298765	1.592500	2.3456	keep	increased d_ffn from 256 to 512
```

Tab-separated. Short 7-char commit hashes. Do not commit results.tsv to git.

## Strategy Hints

Based on the Fibonacci proof-of-concept (26 trainable params on frozen H4 backbone, φ gap 0.025→0.001):

1. **The geometry provides strong inductive bias.** W_nudge naturally converges to rank-1, aligning with 600-cell vertices. Don't fight this — let the nudge stay small.

2. **Chamber utilization matters.** If entropy is low, the model is only using a few chambers. Try: different init for W_nudge, larger top_k, or adding noise to queries during training.

3. **Start with the simplest change.** Learning rate and d_model matter most. Don't change 5 things at once.

4. **The tree is for long sequences.** For seq_len ≤ 256, full attention is faster than tree lookup (Python overhead). Set `use_tree = MAX_SEQ_LEN > 256`.

5. **Watch for mode collapse.** If all heads learn the same nudge direction, the model wastes capacity. Consider adding a diversity loss or different initializations per head.

6. **The golden ratio is not arbitrary.** φ⁻¹ is the most irrational number — golden-angle positions are maximally separated. The positional encoding exploits this. Don't replace it with sinusoidal.

## Data

Currently using character-level text (Shakespeare if available, synthetic Fibonacci-structured text otherwise). To add real data:

1. Download TinyStories: `wget https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean/resolve/main/TinyStories_all_data.tar.gz`
2. Extract to `data/`
3. Update `load_text_data()` in train_cpu.py

For now, the synthetic data is fine for proving the architecture works. Switch to real data once val_bpb is decreasing consistently.

## Phase 6: BitLinear Experiments

Toggle ternary mode with `USE_BITLINEAR = True` in hyperparameters section.

### Experiment Ideas

1. **Baseline comparison**: Same architecture, float vs ternary.
   Measure val_bpb gap. BitNet Reloaded shows <0.1 bpb gap at 100K+ params
   with 2x hidden size.

2. **Hidden size scaling**: If ternary hurts val_bpb, try doubling d_model
   (64->128) while keeping ternary. The 2x scaling law from BitNet Reloaded
   predicts this recovers the gap.

3. **Selective ternary**: Make only the large projections (q/k/v/out) ternary,
   keep the 4x4 nudge layers in float. The nudge is where geometric alignment
   happens --- it might need float precision.

4. **Zero ratio tracking**: Monitor the zero percentage in ternary nudge
   layers across training. High zero% means the head learned feature
   selection (ignoring some Coxeter directions). Plot zero% vs
   chamber_entropy --- they should be inversely correlated.

5. **Chamber preservation sweep**: Run ternary_diagnostics.chamber_preservation
   after each experiment. If preservation drops below 85%, the ternary
   quantization is too aggressive for that architecture.

### Additional Metrics

- `ternary`: yes/no --- whether BitLinear is active
- `chamber_preserve`: mean chamber preservation rate (float vs ternary)
- `mean_zero_pct`: mean zero% across BitLinear layers
- `compression`: weight compression ratio vs float32
- `model_size_kb`: total model size in KB

### Keep/Discard Rules for Ternary

Same as float rules, plus:
- A ternary experiment that matches float val_bpb within 0.05 is a WIN
  (same quality, ~20x smaller weights)
- A ternary experiment with >5% chamber preservation drop is SUSPECT
  (check if val_bpb actually suffered --- preservation can drop without
  hurting quality if the dropped chambers weren't being used)
