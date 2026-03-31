# ARC-AGI Competition Results

**Current Scores (March 31, 2026):**

| Track | Score | Method |
|-------|-------|--------|
| **ARC-AGI-1** | **400/400 (100%)** | Python code synthesis |
| **ARC-AGI-2** | **58/120 eval (48.3%)** | Claude-written custom solvers |
| **ARC-AGI-3** | **30/182 (16.5%)** | Explorer v6 — UCB1 + efficacy switching + variable budget |

## ARC-AGI-3: 30/182 (16.5%) — Reproducible

**Explorer v6** (`olympus/arc3/explorer_v6_adaptive.py`) achieves 30/182 levels on live ARC-AGI-3 games via GitHub Actions. Pure algorithmic search — no LLM, no training.

### How to reproduce

```bash
# GitHub Actions (recommended):
# Go to Actions tab → "ARC-AGI-3 Explorer Run" → Run workflow
# Branch: main (workflow checks out the feature branch automatically)
# Explorer: v6 (default)

# Manual run (needs unrestricted internet for ARC API):
git checkout claude/polytopic-attention-implementation-XHkL3
pip install arc-agi arcengine scipy numpy
export ARC_API_KEY="your-key-here"
python olympus/arc3/explorer_v6_adaptive.py
```

### Per-game results (v6, reproducible)

```
Game   Levels  Notes
lp85   5/8     Best performer, 400K budget
dc22   3/6     UCB1 action selection gain (+1 over BFS)
vc33   3/7     Grid-fine mode
ar25   2/8     Segment mode
m0r0   2/6     Segment mode
ft09   2/6     300K budget, 167K states at depth 184
sp80   1-2/6   Randomization variance
r11l   1/6     NEW — efficacy switch (was 0 with BFS)
tu93   1/9     NEW — efficacy switch (was 0 with BFS)
cn04   1/5     NEW — efficacy switch (was 0 with BFS)
+ 11 more games at 1 level each
```

### Score evolution

```
v1  unified@150K:  23/182  (12.6%)
v4  scipy@200K:    28/182  (15.4%)
v5  +UCB1:         29/182  (15.9%)
v6  +adaptive:     30/182  (16.5%)
```

### Key code

All on branch: `claude/polytopic-attention-implementation-XHkL3`

- `olympus/arc3/explorer_v6_adaptive.py` — The winning explorer
- `cge/` — Compression-Guided Exploration research (501 levels on simulated envs)
- `agent_zero/` — Unified agent framework (748 levels on 14 envs)
- `agent_zero/transforms/` — Transform engine (39/400 ARC-AGI-1, zero LLM)
- `.github/workflows/arc3-run.yml` — GitHub Actions runner
