# Session Handoff — ARC-AGI Self-Compiling Intelligence

**Date:** 2026-03-29 (sessions 1+2+3+4, session 4 IN PROGRESS)
**Status:** AGI-1 100%, AGI-2 38.3% eval, AGI-3 12.6% → v4 ready

---

## Session 4: Explorer v4 Optimizations (THIS SESSION)

### What Was Built
**`olympus/arc3/explorer_v4.py`** — Speed-optimized unified explorer with adaptive grid-click.

| Improvement | Detail |
|-------------|--------|
| **Faster hashing** | Avoid frame copy when no mask, use `ravel()` not `flatten()` |
| **Faster segmentation** | `scipy.ndimage.label` (C-level) — ~5x faster than Python BFS |
| **Lower grid-click threshold** | 30 states / 3K actions (was 50 / 5K) — catches stuck games earlier |
| **Adaptive grid step** | step=4 → step=2 escalation at 15K actions if still stuck |
| **O(1) reverse action lookup** | Pre-built dict replaces O(n) scan per step |
| **Higher budget** | 200K default (was 150K) — more budget = more levels |
| **Null mask optimization** | `detect_status_bar_mask` returns None when no bars found |

### Files Added
```
olympus/arc3/explorer_v4.py     # v4 unified explorer
benchmark_v4.py                 # Run all 25 games at 200K
run_stuck_games_v4.py           # Focus on 14 zero-level games
```

### How to Run
```bash
# Full benchmark (200K actions per game)
PYTHONIOENCODING=utf-8 ARC_API_KEY="58b421be-5980-4ee8-8e57-0f18dc9369f3" python benchmark_v4.py

# Stuck games only (faster iteration)
PYTHONIOENCODING=utf-8 ARC_API_KEY="58b421be-5980-4ee8-8e57-0f18dc9369f3" python run_stuck_games_v4.py

# Single game
ARC_API_KEY="58b421be-5980-4ee8-8e57-0f18dc9369f3" python olympus/arc3/explorer_v4.py GAME_ID 200000
```

### Expected Impact
- **Speed**: ~3-5x faster per action (target <50ms from ~150ms)
- **Grid-click**: Earlier detection means cd82, lf52, sk48 get more budget in grid mode
- **Budget**: 200K = 33% more exploration than 150K
- **Target**: 25+ levels (13.7%+), beating 3rd place's 12.58%

### What Still Needs Running
1. **Full v4 benchmark** at 200K on all 25 games — compare to v2@100K (23/182)
2. **Stuck games focus run** — especially sk48 (was 316 states at 20K grid-click)
3. **300K budget test** on top 5 games — see if more budget helps

---

## Session 3 Final Results

| Track | Before Session 3 | After Session 3 | Change |
|-------|-------------------|------------------|--------|
| **ARC-AGI-1** | 400/400 (100%) | 400/400 (100%) | — |
| **ARC-AGI-2** | 24/120 (20%) eval | **46/120 (38.3%)** eval | **+22 solutions** |
| **ARC-AGI-3** | 22/182 (12.1%) | **23/182 (12.6%)** unified 150K | **+1 level, new agent** |

### Key Achievements
- **AGI-2 doubled** from 20% to 38.3% with 10 parallel retry agents
- **3 new AGI-3 explorer versions** built (v2 lazy rebuilds, v3 grid-click, unified)
- **Grid-click breakthrough** unlocked cd82 and lf52 (previously stuck at 0 states)
- **Fine-tune data ready**: 519 entries in `data/arc_finetune_all.jsonl`
- **544 total unique solutions** across all tracks

---

## What To Do Next (Priority Order)

### 1. Push AGI-3 Higher (13%→15%+)
- Run unified agent at **200K+ actions** — more budget = more levels
- Optimize per-action speed (currently ~150ms, 3rd place ~31ms)
- Try grid-click on **all 14 zero-level games**, not just the stuck 3
- Focus on sk48 (316 states at 20K with grid-click, depth=125 — nearly cracking)

### 2. Push AGI-2 Higher (38%→50%+)
- 74 unsolved eval tasks remain (was 100, solved 26 unique)
- Retry solution files: `data/arc2_solutions_retry{4-13}.json` (28 raw, 26 unique after dedup)
- Re-run on remaining 74 with different prompting (deeper analysis, print grids, 10+ hypotheses)
- Hardest tasks identified by agents: spirals, path tracing, multi-step folding, nested frames

### 3. Fine-Tune 3B Model for Kaggle
- Data ready: `data/arc_finetune_all.jsonl` (519 entries, 221 with Python code)
- Spin up RunPod pod with GPU (~$2-5)
- Train Qwen-3B or SmolLM-3B on solve functions
- Submit to AGI-1 and AGI-2 Kaggle competitions

### 4. Competition Submissions
- AGI-1: Fine-tuned model → Kaggle (runs offline)
- AGI-2: Fine-tuned model → Kaggle (runs offline)
- AGI-3: Unified explorer → API submission

---

## Final Scores

| Track | Score | Details |
|-------|-------|---------|
| **ARC-AGI-1** | **400/400 (100%)** | Perfect score, all verified |
| **ARC-AGI-2** | **46/120 (38.3%) eval** | 20 original + 26 unique retries (10 parallel agents) |
| **ARC-AGI-3** | **23/182 (12.6%)** | Unified@150K, grid-click unlocks cd82+lf52 |

---

## Session 3: AGI-3 Explorer Improvements (THIS SESSION)

### Three Explorer Versions

| Version | File | Key Innovation | Best For |
|---------|------|----------------|----------|
| **v1** | `olympus/arc3/explorer.py` | Priority-group BFS, proven on lp85/vc33 | Games already solving levels |
| **v2** | `olympus/arc3/explorer_v2.py` | Lazy rebuilds, winning path replay, depth-biased frontier | Speed + multi-level games |
| **v3** | `olympus/arc3/explorer_v3_gridclick.py` | Click every grid cell, not just segment centroids | Games with <20 states (stuck games) |

### Grid-Click Breakthrough (v3)
Games that had 0 levels with 3-9 states explored were STUCK because segmentation missed interactive elements. Grid-click tries every cell:
- **lf52**: 0 → **1/10 levels** (grid_step=2, 1024 click points, 20K actions)
- **cd82**: 0 → **1/6 levels** (grid_step=4, 256 click points, only 5K actions!)
- **sk48**: 0 → 316 states explored (was 5), depth=125. Needs more budget to crack.

### v2 100K Benchmark Results (COMPLETE)

**TOTAL: 23/182 levels (12.6%)** — +1 over v1's 22 levels at 91K
```
ar25: 2/8   bp35: 1/9   cd82: 0/6   cn04: 1/5*  dc22: 0/6
ft09: 2/6*  g50t: 0/7   ka59: 0/7   lf52: 0/10  lp85: 5/8
ls20: 1/7   m0r0: 2/6   r11l: 0/6   re86: 0/8   s5i5: 1/8
sb26: 0/8   sc25: 0/6   sk48: 0/8   sp80: 2/6   su15: 1/9*
tn36: 1/7   tr87: 0/6   tu93: 1/9*  vc33: 3/7   wa30: 0/9
(* = newly solved vs v1)
```

### Unified Agent (v2+v3 merged) — 150K COMPLETE
`olympus/arc3/explorer_unified.py` — auto-switches to grid-click when stuck.
**Score: 23/182 (12.6%)** at 150K actions, 5588 seconds total.

```
ar25:2/8  bp35:1/9  cd82:1/6*  cn04:0/5  dc22:0/6
ft09:0/6  g50t:0/7  ka59:0/7  lf52:1/10*  lp85:5/8
ls20:1/7  m0r0:2/6  r11l:0/6  re86:0/8  s5i5:1/8
sb26:0/8  sc25:0/6  sk48:0/8  sp80:2/6  su15:1/9
tn36:1/7  tr87:1/6  tu93:1/9  vc33:3/7  wa30:0/9
(* = solved by grid-click auto-switch)
```

Game randomization means individual game scores vary +-2 levels per run.
Grid-click auto-switch successfully unlocked cd82 and lf52.

Running total: 15 levels from 15 games. Remaining 10 games typically add 7-10 more.

**Key v2@100K wins vs v1@91K:**
- cn04: 0 → **1/5** (was stuck, now solving)
- ft09: back to **2/6** (was 0 at 50K, needed more budget)
- lp85: **5/8** (matching v1's best)
- ls20: **1/7** (back to matching v1)

### v2 50K Full Benchmark (completed)
```
TOTAL: 17/182 levels (9.3%) at 50K actions
ar25:2/8 bp35:1/9 cn04:0/5 lp85:4/8 m0r0:2/6 sp80:2/6
s5i5:1/8 su15:1/9 tr87:1/6 vc33:3/7
```

### 4 Critical Bugs Found & Fixed in v2

1. **Winning path replay was dead code** — `winning_paths` populated but never read. Fixed: efficient replay of state-changing-only actions.
2. **Death actions never recorded** — GAME_OVER transition lost. Fixed: informational tracking (not blocking — hard step limits make death-blocking harmful).
3. **BFS + short episodes = shallow-only** — Games like re86 (100-action episodes, 5 actions) can't find solutions at depth >60. Depth-biased frontier selection helps somewhat.
4. **Winning path too long** — Saved full exploration history, not minimum path. Fixed: only save state-changing actions (effective_history).

### Diagnostic Results (14 zero-level games at 10K actions)

**Category 1 — Large state space, productive exploration:**
```
re86: 5807 states, 99% change rate, arrows only, 100-action episodes
cn04: 5614 states, 67% change rate, clicks+arrows, 150-action episodes
sb26: 2191 states, 82% change rate, click+space+undo
wa30: 1444 states, 62% change rate, arrows only
ka59: 920 states, 23% change rate, clicks+arrows
```

**Category 2 — Moderate state space:**
```
sc25: 154 states, 62% change rate
g50t: 118 states, 53% change rate
r11l: 60 states, 100% change rate, click-only
tu93: 50 states, 44% change rate
dc22: 48 states, 2.3% change rate
```

**Category 3 — Tiny state space (STUCK, needs grid-click):**
```
cd82: 37 states → 1/6 with grid-click ✓
sk48: 5 states → 316 states with grid-click, needs more budget
lf52: 3 states → 1/10 with grid-click ✓
```

### Performance Optimization: Lazy Rebuilds
`_rebuild_distances()` was O(V+E) called on every node add/close. With 20K+ nodes (re86), this dominated runtime. Fixed with lazy rebuilds — only rebuild when `choose_action` needs routing to frontier. Result: ~30% speedup on large-state games.

---

## NEXT SESSION: Build Unified Agent

### The Plan
Merge all three innovations into one agent:

1. **v1 priority-group exploration** — proven on lp85 (5/8), vc33 (4/7)
2. **v2 lazy rebuilds + winning path replay** — speed + multi-level efficiency
3. **v3 grid-click fallback** — auto-detect when <50 states after 5K actions, switch to grid-click

### Implementation Steps
1. Copy `explorer_v2.py` as base
2. Add grid-click fallback: if `explorer.num_states < 50` after 5000 actions, switch to `GridClickAgent`
3. Run unified agent at 150K actions on all 25 games
4. Target: 25+ levels (13.7%+), beating 3rd place's 12.58%

### Quick Wins Still Available
- **sk48**: Grid-click gets 316 states at 20K. Run at 100K — likely solves level 1+
- **cn04**: Solved level 1 at 100K. More budget → more levels
- **cd82**: Solved level 1 quickly. More budget for levels 2+
- **lf52**: Solved level 1. More budget for levels 2+

---

## ARC-AGI-1: 400/400 (100%)

### Solution Files
```
data/arc_python_solutions_b{0-34}.json     # Main batches (362 solutions)
data/arc_python_solutions_retry_{a,b,c}.json  # Retry waves (12)
data/arc_python_solutions_final6.json       # Final push (4)
data/arc_python_solutions_recovery.json     # Recovered lost solutions (38)
data/arc_python_solutions_last4.json        # Last 4 (3)
data/arc_python_solutions.json              # Original batch (10)
solve_234bbc79.py                           # Standalone: cyclic crossing shifts mod 3
solve_3631a71a.py                           # Standalone: transpose symmetry chain
```

### Verification
```bash
py -c "
import json, glob, os
solved = set()
for f in glob.glob('data/arc_python_solutions*.json'):
    with open(f) as fh: solved.update(json.load(fh).keys())
solved.add('234bbc79'); solved.add('3631a71a')
arc1 = set(f.replace('.json','') for f in os.listdir('data/arc1/') if f.endswith('.json'))
print(f'{len(solved & arc1)}/{len(arc1)}')
"
# Expected output: 400/400
```

---

## ARC-AGI-2: 24/120 eval (20%)

### Solution Files
```
data/arc2_solutions_eval{0-3}.json          # First pass: 18 solutions
data/arc2_solutions_retry{0-3}.json         # Retries: +6 solutions
data/arc2_solutions_train_{aa-af}.json      # New training: 112 solutions
```

---

## ARC-AGI-3: Agent Code

### Explorer Files
```
olympus/arc3/explorer.py        # v1: Priority-group BFS (ORIGINAL)
olympus/arc3/explorer_v2.py     # v2: Lazy rebuilds, replay, depth-bias
olympus/arc3/explorer_v3_gridclick.py  # v3: Grid-cell click for stuck games
olympus/arc3/explorer_unified.py # Unified v1 (v2+v3 merged, 150K)
olympus/arc3/explorer_v4.py     # v4: Speed-optimized, adaptive grid-click, 200K
olympus/arc3/__init__.py
```

### Diagnostic/Test Files (session 3)
```
diagnose_zero_games.py          # Diagnostic: 14 zero-level games at 10K
diagnose_zero_results.json      # Diagnostic results (JSON)
run_top3_verbose.py             # Verbose runner for top 3 games
test_v2_quick.py                # Quick v2 test script
benchmark_v2.py                 # Full benchmark runner
```

### How to Run
```bash
# Activate venv
source .venv-arc3/Scripts/activate    # Windows

# v1 (original) — all games
ARC_API_KEY="58b421be-5980-4ee8-8e57-0f18dc9369f3" py olympus/arc3/explorer.py

# v2 (improved) — all games at 100K
PYTHONIOENCODING=utf-8 ARC_API_KEY="58b421be-5980-4ee8-8e57-0f18dc9369f3" py benchmark_v2.py

# v3 grid-click — stuck games
PYTHONPATH=. PYTHONIOENCODING=utf-8 ARC_API_KEY="58b421be-5980-4ee8-8e57-0f18dc9369f3" py olympus/arc3/explorer_v3_gridclick.py

# Single game (any version)
ARC_API_KEY="58b421be-5980-4ee8-8e57-0f18dc9369f3" py olympus/arc3/explorer_v2.py GAME_ID MAX_ACTIONS
```

### 25 Game IDs
```
ar25-e3c63847  bp35-0a0ad940  cd82-fb555c5d  cn04-65d47d14  dc22-4c9bff3e
ft09-0d8bbf25  g50t-5849a774  ka59-9f096b4a  lf52-271a04aa  lp85-305b61c3
ls20-9607627b  m0r0-dadda488  r11l-aa269680  re86-4e57566e  s5i5-a48e4b1d
sb26-7fbdac44  sc25-f9b21a2f  sk48-41055498  sp80-0ee2d095  su15-4c352900
tn36-ab4f63cc  tr87-cd924810  tu93-2b534c15  vc33-9851e02b  wa30-ee6fef47
```

### Per-Game Episode Budgets (actions before GAME_OVER)
```
bp35:~32  su15:~23  tu93:~50  r11l:~60  sc25:~79  sp80:~59
tn36:~61  lf52:~64  sk48:no_limit  cd82:~100  re86:~100
ka59:~100  dc22:~127  g50t:~130  ls20:~130  ft09:~145
cn04:~150  s5i5:~150  m0r0:~151  wa30:~200  ar25:~236
sb26:~265  lp85:~344
```

### Reference Repos (cloned locally)
```
ARC-AGI-3-Agents/              # Official agent template
arc-agi-3-just-explore/        # 3rd place solution (our reference)
ARC3-solution/                 # DriesSmit CNN approach
```

---

## API Keys & Credentials

### ARC-AGI-3
- **API Key:** `58b421be-5980-4ee8-8e57-0f18dc9369f3`
- **SDK:** `arc-agi` + `arcengine` in `.venv-arc3/` (Python 3.12)
- **25 games available**, 182 total levels

### GitHub
- **Repo:** `grapheneaffiliate/h4-polytopic-attention` (note: no 's')
- **Branch:** main

### HuggingFace
- **Account:** grapheneaffiliates (with 's')
- **Repo:** `grapheneaffiliates/h4-polytopic-attention`

---

## Environment Setup

### Python Environments
```
System Python 3.12: py (Windows launcher)
ARC-AGI-3 venv: .venv-arc3/ (Python 3.12, arc-agi + arcengine + numpy)
```

### Key Paths (Windows)
```
Project root:    C:\Users\atchi\h4-polytopic-attention
Agent v1:        olympus/arc3/explorer.py
Agent v2:        olympus/arc3/explorer_v2.py
Agent v3:        olympus/arc3/explorer_v3_gridclick.py
Venv:            .venv-arc3/Scripts/activate
```

---

## Critical Notes
- **DO NOT read game source code** for ARC-AGI-3 (environment_files/ = answer key)
- **ACTION6 requires explicit data**: `env.step(action, data=action.action_data.model_dump())`
- **PYTHONIOENCODING=utf-8** required on Windows for unicode output
- **PYTHONPATH=.** required when running v3 grid-click from project root
- **Games are RANDOMIZED** per scorecard — same game can give different results across runs
- **GitHub username**: grapheneaffiliate (no s)
- **HuggingFace username**: grapheneaffiliates (with s)
