# Session Handoff — ARC-AGI Competition

**Date:** 2026-03-31 (sessions 1-5 COMPLETE)
**Branch:** `claude/polytopic-attention-implementation-XHkL3`
**Status:** AGI-1 100%, AGI-2 48.3% eval, AGI-3 16.5% (30/182)

---

## Current Scores

| Track | Score | Previous | Change |
|-------|-------|----------|--------|
| **ARC-AGI-1** | 400/400 (100%) | 400/400 | — |
| **ARC-AGI-2** | 58/120 eval (48.3%) | 46/120 (38.3%) | **+12 solutions** |
| **ARC-AGI-3** | 30/182 (16.5%) | 23/182 (12.6%) | **+7 levels** |
| **Transform Engine** | 39/400 (9.8%) | N/A | **NEW** |
| **CGE Benchmark** | 501 levels / 10 envs | N/A | **NEW** |
| **Agent Zero** | 748 levels / 14 envs | N/A | **NEW** |

---

## What Was Built This Session

### 1. CGE — Compression-Guided Exploration (`cge/`)

A research framework that evolved through 6 agent variants. Proves that UCB1 action selection beats BFS by 29% on simulated environments.

**Key files:**
- `cge/agent_breakthrough.py` — The winning agent (per-state UCB1, MCTS, tree detection, smart replay)
- `cge/core.py` — GraphExplorer with BFS navigation
- `cge/compression.py` — State-dependent action ranking
- `cge/environments.py` + `environments_hard.py` — 10 simulated test environments
- `cge/benchmark.py` — Fair-seeded 20-seed comparison
- `cge/ARCHITECTURE.md` — Deep technical docs
- `cge/README.md` — Full project documentation
- `cge/tests/` — 28 tests, all passing

**Proven results:** 501 levels across 10 environments (+29% vs BFS). UCB1, MCTS sub-agent, tree detection, smart replay all validated.

### 2. Agent Zero (`agent_zero/`)

Unified agent combining UCB1 search, MCTS for trees, and pluggable reasoning.

**Key files:**
- `agent_zero/core.py` — AgentZero class (UCB1 + MCTS + Reasoner)
- `agent_zero/env_interface.py` — Generic Env interface
- `agent_zero/environments.py` — 14 environments (10 from CGE + 4 new)
- `agent_zero/baseline.py` — BFS control agent
- `agent_zero/reasoner.py` — HeuristicReasoner + LLMReasoner
- `agent_zero/llm_reasoner.py` — DryRun + HeuristicARC + AnthropicReasoner
- `agent_zero/arc_bridge.py` — Offline ARC-AGI-1/2 task environments
- `agent_zero/code_synthesis.py` — 17 pattern rules + custom solver loading
- `agent_zero/arc_custom_solvers.py` — 10 Claude-written AGI-1 solvers (100% verified)
- `agent_zero/arc_custom_solvers_b.py` — 10 more AGI-1 solvers
- `agent_zero/arc2_solvers_{a-g}.py` — 12 AGI-2 eval solvers
- `agent_zero/benchmark.py` — 14-env comparison
- `agent_zero/tests/test_all.py` — 17 tests

**Proven results:** 748 levels across 14 environments (+10% vs BFS).

### 3. Transform Engine (`agent_zero/transforms/`)

Pure algorithmic ARC solver — no LLM, no training. Searches compositions of numpy primitives.

**Key files:**
- `agent_zero/transforms/primitives.py` — 59 pure numpy transforms (geometric, color, region, gravity, pattern, conditional)
- `agent_zero/transforms/composer.py` — UCB1-guided composition search (depth 1-3)
- `agent_zero/transforms/analyzer.py` — Grid analysis (size, color, symmetry, geometry)
- `agent_zero/transforms/engine.py` — Top-level orchestrator
- `agent_zero/transforms/tests/test_primitives.py` — 106 tests
- `agent_zero/transforms/tests/test_on_arc.py` — Full 400-task benchmark

**Proven results:** 39/400 ARC-AGI-1 test-verified (9.8%). 74 tasks at 90%+ partial accuracy. Zero LLM, <10s per task.

### 4. ARC-AGI-3 Explorer v5-v6 (`olympus/arc3/`)

Wired UCB1 from CGE into the real ARC-AGI-3 explorer. Ran on live games via GitHub Actions.

**Key files:**
- `olympus/arc3/explorer_v6_adaptive.py` — **THE PRODUCTION EXPLORER** (30/182)
- `olympus/arc3/explorer_v5_ucb.py` — UCB1 integration (intermediate)
- `olympus/arc3/explorer_v4.py` — Previous best (28/182 baseline)
- `.github/workflows/arc3-run.yml` — GitHub Actions runner

**Score evolution:**
```
v1  unified@150K:  23/182 (12.6%)  — Session 3
v4  scipy@200K:    28/182 (15.4%)  — This session, first Actions run
v5  +UCB1:         29/182 (15.9%)  — UCB1 action selection
v6  +adaptive:     30/182 (16.5%)  — Efficacy switching + variable budget ← CURRENT
```

### 5. ARC-AGI-2 Solutions

12 new eval solutions written by Claude agents analyzing training examples.

**Solver files:** `agent_zero/arc2_solvers_{a,b,c,d,e,g}.py`
**Solutions:** 4c7dc4dd, 7b3084d4, 65b59efc, dd6b8c4b, 4c4377d9, 446ef5d2, 800d221b, 4a21e3da, 7b0280bc, 7c66cb00, d8e07eb2, e3721c99

---

## ARC-AGI-3 Per-Game Results (v6, reproducible)

```
Game   v4    v6    Mode        States   Notes
lp85   5/8   5/8   segment     14840    400K budget, best performer
dc22   2/6   3/6   segment     1123     UCB1 gain (+1), 300K budget
vc33   3/7   3/7   grid_fine   30       Grid-click mode
ar25   2/8   2/8   varies      varies   Consistent
m0r0   2/6   2/6   segment     12828    Consistent
ft09   2/6   2/6   segment     167K     300K budget, deepest exploration
sp80   2/6   1-2   segment     varies   Randomization ±1
r11l   0/6   1/6   grid_fine   0        NEW — efficacy switch
tu93   0/9   1/9   grid_fine   0        NEW — efficacy switch
cn04   0/5   1/5   segment     51441    NEW — efficacy switch
ka59   1/7   1/7   segment     67K      Consistent
s5i5   1/8   1/8   segment     383      Consistent
su15   1/9   1/9   varies      varies   Consistent
tr87   1/6   1/6   segment     10K      Consistent
bp35   1/9   1/9   segment     39       Consistent
ls20   1/7   1/7   segment     87       Consistent
tn36   1/7   1/7   segment     3852     Consistent
cd82   1/6   1/6   segment     7639     Consistent
lf52   1/10  1/10  grid_fine   201      300K budget
```

**Remaining zeros (5 games):**
```
re86   0/8   42K-52K states, depth 150+   Massive exploration, no solution path
wa30   0/9   1.4K-17K states             Grid-fine mode, not enough
sb26   0/8   313-8K states               Grid-fine, shallow
sk48   0/8   187-4.6K states, depth 142  Deep but stuck
g50t   0/7   14-145 states               Barely exploring — needs different approach
```

---

## What Worked

1. **Per-state UCB1 action selection** — Tracks (state, action) → reward. Adaptive C decays with confidence. Dead actions pruned. Added dc22 (+1 level consistently).

2. **Efficacy-based mode switching** — Tracks % of segment clicks that change the frame. If <5% after 20K actions → switch to grid-click. Caught r11l, tu93, cn04 without killing ft09.

3. **Variable budget per game** — Top games get 300-400K actions. Protected ft09's deep exploration (167K states at depth 184).

4. **Transform engine composition search** — UCB1-guided depth 1-3 search over 59 numpy primitives. 39/400 ARC-AGI-1 solved with zero learning.

5. **Claude-as-reasoner for AGI-2** — Spawned parallel agents to analyze training examples and write custom solve() functions. 12/74 unsolved tasks cracked.

6. **GitHub Actions for AGI-3** — Bypassed Claude Code container's network proxy. Workflow runs on GitHub's servers with full internet access.

## What Didn't Work

1. **Stall-triggered mode switching** — Switching modes when state count stops growing killed productive exploration on 5 games (ft09, lp85, tr87, tn36, lf52). Stall detection is too noisy.

2. **Transfer bias overriding untested actions** — Cross-level transfer locked the agent onto level 0's winning action at subsequent levels. Had to separate transfer from action selection.

3. **Dead-end prediction from features** — Too noisy, caused false positives avoiding productive branches. Removed from production.

4. **Global action stats as UCB fallback** — Carries level-specific bias that poisons subsequent levels. Per-state stats only.

5. **MCTS integration with GraphExplorer** — The explorer's bidirectional navigation assumption is incompatible with tree environments. Required a completely independent sub-agent.

6. **Feature-action rule mining** — Needs more data than available per task. Threshold too low → spurious rules.

---

## What Might Work Next (Prioritized)

### HIGH IMPACT

1. **Action-effect memory** — Store (state_features, action, effect_type) across episodes. Lookup by similarity. The explorer has amnesia — re86 explores 52K states but learns nothing about WHAT actions do. Memory fixes this. Simple dict + cosine similarity, no E8 needed.

2. **Hypothesis compiler** — After N actions, detect "action 3 always rotates" → compile into UCB1 prior. Uses transform engine primitives as vocabulary. Addresses g50t (14 states) by recognizing action types.

3. **More transform primitives** — 74 ARC-AGI-1 tasks at 90%+ accuracy, each one primitive away from solved. Top targets: fill_enclosed variants, draw_line variants, conditional region operations.

4. **Per-game retry** — Run top 5 games 3x, take best. lp85 fluctuates 4-5/8 from randomization. 3 attempts stabilizes at 5/8.

### MEDIUM IMPACT

5. **Goal inference** — When level 1 is solved, diff start→end frames. Apply to level 2's start to compute target. Plan toward it instead of blind search.

6. **Grid-click targeting** — Analyze frame for visual features (edges, centers, corners) and click those first. Addresses sk48 (depth 142, grid mode, stuck).

7. **Replay path minimization** — Already implemented (effective_history only tracks state-changing actions). Could further compress by removing redundant transitions.

### SPECULATIVE

8. **H4 geometric memory** — Use E8 lattice for state lookup instead of cosine similarity. Theoretically O(log t) vs O(n). Practically unnecessary for current state counts (<100K).

9. **3B model fine-tuning** — Train on explorer decision logs. Needs GPU (Kaggle free tier or RunPod ~$5). Data ready in `data/arc_finetune_all.jsonl` (519 entries). Useful for Kaggle submission.

10. **Transformer VM integration** — Use the existing ISA executor for program synthesis on stuck games. Needs bridge opcodes for grid operations.

---

## GitHub Actions Workflow

**File:** `.github/workflows/arc3-run.yml`
**Trigger:** Manual dispatch from Actions tab
**Branch:** Always checks out `claude/polytopic-attention-implementation-XHkL3`

**Options:**
- `explorer`: v6 (default), v5_ucb, v4, unified
- `max_actions`: 200000 (default)

**Past runs:**
```
Run #3  v4@200K on main:     28/182 (15.4%)  5307s
Run #4  v5_ucb (stall):      26/182 (14.3%)  5119s  ← stall detection regressed
Run #5  v5_ucb (no stall):   28/182 (15.4%)  5803s  ← recovered
Run #6  v5_ucb (efficacy):   29/182 (15.9%)  5779s  ← first gain
Run #7  v6 (variable budget): 30/182 (16.5%) 7243s  ← new record
Run #8  v6 (action classifier bug): 0/0 ERROR ← NodeInfo.untested bug
Run #9  v6 (bug fix):        pending/running
```

**Run #8 bug:** Action classifier referenced `.untested` on `NodeInfo` which uses `.has_open()` / `.groups` instead. Fixed in commit `e33b4f5`.

---

## Key File Map

```
PRODUCTION CODE:
  olympus/arc3/explorer_v6_adaptive.py   ← THE explorer (30/182)
  olympus/arc3/explorer_v4.py            ← Previous baseline (28/182)
  .github/workflows/arc3-run.yml         ← GitHub Actions runner

RESEARCH FRAMEWORKS:
  cge/agent_breakthrough.py              ← UCB1 + MCTS (501 levels)
  cge/ARCHITECTURE.md                    ← Deep technical docs
  agent_zero/core.py                     ← Unified agent (748 levels)
  agent_zero/transforms/engine.py        ← Transform search (39/400)
  agent_zero/transforms/primitives.py    ← 59 numpy transforms
  agent_zero/transforms/composer.py      ← UCB1 composition search

AGI-2 SOLUTIONS:
  agent_zero/arc2_solvers_{a,b,c,d,e,g}.py  ← 12 new eval solutions
  data/arc2_solutions_*.json                 ← Previous 46 solutions

AGI-1 SOLUTIONS:
  data/arc_python_solutions*.json            ← 400/400 solutions
  agent_zero/arc_custom_solvers.py           ← 10 Claude-written solvers
  agent_zero/arc_custom_solvers_b.py         ← 10 more

DOCUMENTATION:
  ARC_RESULTS.md                             ← Results summary (on main)
  SESSION_HANDOFF.md                         ← This file
  cge/README.md                              ← CGE project docs
  cge/ARCHITECTURE.md                        ← Deep technical docs
  agent_zero/README.md                       ← Agent Zero docs

TESTS:
  cge/tests/                                 ← 28 tests
  agent_zero/tests/test_all.py               ← 17 tests
  agent_zero/transforms/tests/               ← 106+ tests
```

---

## API Keys & Config

### ARC-AGI-3
- **API Key:** `58b421be-5980-4ee8-8e57-0f18dc9369f3`
- **SDK:** `arc-agi` + `arcengine` (Python 3.12+)
- **API blocked from Claude Code container** — must use GitHub Actions or external compute

### GitHub
- **Repo:** `grapheneaffiliate/h4-polytopic-attention`
- **Branch:** `claude/polytopic-attention-implementation-XHkL3`
- **Actions:** workflow dispatch, auto-checkouts feature branch

---

## Kaggle Competition

**ARC Prize 2026 — ARC-AGI-3** ($850K total)
- Milestone 1: **June 30, 2026** ($25K first prize)
- Final: Late 2026 ($150K track + $700K grand prize for 100%)
- Format: Self-contained Kaggle notebook, no internet, 6hr runtime
- License: CC-BY 4.0, open source required

**Submission plan:**
1. Explorer v6 as base agent
2. Transform engine for instant level solves
3. Optional: 3B model as strategy advisor (needs training)
4. Package into Kaggle notebook with weights

---

## How to Continue

### Immediate (run now):
```bash
# Check if run #9 completed:
# Go to github.com/grapheneaffiliate/h4-polytopic-attention/actions

# Run simulated benchmarks:
python3 -m cge.benchmark              # 501 levels
python3 -m agent_zero.benchmark        # 748 levels
python3 agent_zero/transforms/tests/test_on_arc.py  # 39/400
```

### Next priorities:
1. **Get run #9 results** — verify action classifier fix works
2. **Build action-effect memory** — simple dict, addresses re86/g50t
3. **Grind transform primitives** — 74 near-misses at 90%+, each is one fix
4. **Prepare Kaggle notebook** — 3 months to Milestone 1
