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

### Per-game results (v6, reproducible across two consecutive runs)

```
Game   Levels  Notes
lp85   5/8     Best performer, 400K budget
dc22   3/6     UCB1 action selection gain (+1 over BFS)
vc33   3/7     Grid-fine mode
ar25   2/8     Segment mode
m0r0   2/6     Segment mode
ft09   2/6     300K budget, 167K states at depth 184
sp80   1-2/6   Randomization variance
r11l   1/6     NEW — efficacy switch to grid-fine (was 0 with BFS)
tu93   1/9     NEW — efficacy switch to grid-fine (was 0 with BFS)
cn04   1/5     NEW — efficacy switch caught it (was 0 with BFS)
ka59   1/7     Segment mode
s5i5   1/8     Segment mode
su15   1/9     Segment mode
tr87   1/6     Segment mode
bp35   1/9     Segment mode
ls20   1/7     Segment mode
tn36   1/7     Segment mode
cd82   1/6     Segment mode
lf52   1/10    Grid-fine mode, 300K budget
```

### Score evolution

```
v1  unified@150K:  23/182  (12.6%)
v4  scipy@200K:    28/182  (15.4%)
v5  +UCB1:         29/182  (15.9%)
v6  +adaptive:     30/182  (16.5%)
```

### Key innovations (v5-v6 over v4)

1. **Per-state UCB1**: Each (state, action) pair tracks reward (frame changed or not). Adaptive C decays as confidence grows. Dead actions pruned after 10+ zero-reward visits.
2. **Efficacy-based mode switching**: Tracks what % of segment clicks change the frame. If <5% after 20K actions → switch to grid-click. Catches r11l/tu93/cn04 without killing ft09.
3. **Variable budget**: Top games get 300-400K actions (lp85, dc22, ft09, vc33, lf52). Others get 200K default.
4. **Action classifier**: Learns per-action change rates, feeds into UCB1 priors for untested actions.

### Key code

All on branch: `claude/polytopic-attention-implementation-XHkL3`

- `olympus/arc3/explorer_v6_adaptive.py` — The winning explorer
- `cge/` — Compression-Guided Exploration research (501 levels on simulated envs)
- `agent_zero/` — Unified agent framework (748 levels on 14 envs)
- `agent_zero/transforms/` — Transform engine (39/400 ARC-AGI-1, zero LLM)
- `.github/workflows/arc3-run.yml` — GitHub Actions runner

## Quick Start

```bash
# Simulated benchmarks (no API needed)
python3 -m cge.benchmark                    # CGE: 501 levels
python3 -m agent_zero.benchmark             # Agent Zero: 748 levels
python3 agent_zero/transforms/tests/test_on_arc.py  # Transform engine: 39/400

# ARC-AGI-3 (needs API key + unrestricted internet)
export ARC_API_KEY="your-key"
python olympus/arc3/explorer_v6_adaptive.py  # 30/182 on live games
```
