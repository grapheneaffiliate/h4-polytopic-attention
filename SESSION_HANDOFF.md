# Session Handoff — ARC-AGI-3 Local Solving

**Date:** 2026-04-02 (Session 6+7 — LOCAL SOLVING + ABSTRACT SOLVERS)
**Branch:** `work/harness-pattern`
**Goal:** 100% on ARC-AGI-3 (182/182 levels across 25 games) via pure CPU algorithms

---

## Current Local Solving Results

| Game | Explorer v6 | Local Solver | Method | Status |
|------|-------------|--------------|--------|--------|
| **lp85** | 5/8 | **8/8** | Abstract permutation BFS (0.2s) | DONE |
| **tn36** | 1/7 | **7/7** | Abstract opcode solver (0.4s) | DONE |
| **tu93** | 1/9 | **5/9** | Frame-hash BFS (deepcopy) | Levels 0-4 |
| **ft09** | 2/6 | **3/6** | GF(p) constraint solver | Levels 0-2 |
| **sp80** | 2/6 | **1/6** | Generic BFS (deepcopy) | Level 0 only |
| **cd82** | 1/6 | **1/6** | Generic BFS (deepcopy) | Level 0 only |
| **ls20** | 1/7 | **1/7** | Generic BFS (deepcopy) | Level 0 (13 actions) |
| dc22 | 3/6 | 0 | BFS too slow | |
| vc33 | 3/7 | 0 | Click game, analysis done | |
| ar25 | 2/8 | 0 | Analysis done | |
| m0r0 | 2/6 | 0 | Not analyzed | |
| r11l | 1/6 | 0 | Click game, deepcopy works | |
| cn04 | 1/5 | 0 | Not analyzed | |
| ka59 | 1/7 | 0 | Not analyzed | |
| s5i5 | 1/8 | 0 | Not analyzed | |
| su15 | 1/9 | 0 | Not analyzed | |
| tr87 | 1/6 | 0 | Rotation puzzle, 9K states explored, too deep | |
| bp35 | 1/9 | 0 | Not analyzed | |
| sc25 | 0/6 | 0 | Deepcopy BROKEN (closures), spell-casting | |
| sb26 | 0/8 | 0 | Tile matching, 1744 states too shallow | |
| lf52 | 1/10 | 0 | Not analyzed | |
| re86 | 0/8 | 0 | Not analyzed | |
| wa30 | 0/9 | 0 | Not analyzed | |
| sk48 | 0/8 | 0 | Not analyzed | |
| g50t | 0/7 | 0 | Deepcopy BROKEN, grid puzzle, analysis done | |

**Local total: 26/182** (14.3%)
**Combined unique (explorer + local): ~42/182** (23.1%)

---

## Critical Discovery: Deepcopy Closure Bug

**Python `copy.deepcopy()` breaks games that store lambda closures referencing `self`.**

Affected games: **TN36, G50T, SC25** (verified)
Working games: TU93, CD82, TR87, LS20, SB26, R11L, AR25, SP80, DC22

**Root cause:** Lambdas like `lambda: self.move(0, 4)` in opcode dictionaries capture the original `self` reference. After deepcopy, the lambda still calls the ORIGINAL object.

**Solutions:**
1. **Reset + replay** (correct, slower): recreate game from scratch for each BFS state
2. **Abstract solvers** (fastest): bypass game engine entirely, BFS on pure tuples
3. **Fix closures after deepcopy** (not implemented)

---

## Infrastructure Built

### scripts/local_runner.py — Game Loading API
- `load_game_class(game_id)` → loads Python source from environment_files/
- `create_game(cls)` → creates and resets game instance
- `step_game(game, action)` → performs one action
- `get_valid_actions(game)` → returns all valid actions
- Works with `.venv-arc3` (Python 3.12 + arcengine)

### scripts/solver_fast.py — Generic BFS
- Frame-hash deduplication (correct for all games)
- `enumerate_actions()` — keyboard + sys_click + grid clicks
- deepcopy for state branching (~30-60 states/s)
- Results: TU93 5/9, CD82 1/6, LS20 1/7, SP80 1/6

### scripts/solve_lp85.py — LP85 Abstract Solver
- Extracts button permutations from source code
- BFS on pure tuples at 23,584 states/s
- 8/8 in 0.2 seconds

### scripts/solve_tn36.py — TN36 Abstract Solver
- Computes opcodes algebraically from start/target state
- Uses reset+replay (not deepcopy) to verify
- 7/7 in 0.4 seconds

### scripts/solve_ft09.py — FT09 Constraint Solver
- GF(p) Gaussian elimination for Lights-Out constraints
- Null space search for NOT_EQUAL constraints
- 3/6 levels (remaining need 3-color GF(3) extension)

### scripts/execute_solutions.py — API Replay
- Reads pre-computed solutions from solutions/*.json
- Replays action sequences via ARC-AGI API

---

## Game Analysis Completed

### Games with Abstract Solver Potential

**LP85** (DONE 8/8): Button permutations → BFS on tuple state
**TN36** (DONE 7/7): Opcode effects → algebraic program computation
**FT09** (3/6): Lights-Out → GF(p) linear algebra (need GF(3) for remaining)
**G50T** (0/7): Grid movement + obstacle. Player at (13,7) → goal at (43,49). Deepcopy broken. Need abstract state: (player_x, player_y, move_counter, obstacle_x). Undo mechanic (ACTION5).
**SC25** (0/6): Spell-casting. Deepcopy broken. 3x3 spell grid (512 combos) + player movement. Spells: teleport, size change, fireball. Win: reach exit.
**TR87** (0/6): Rule rotation puzzle. 7 rotation variants per sprite. Very deep solutions (37+ baseline). Need constraint solver approach.

### Games Needing Source Analysis

**R11L**: Click-drag legs to match target positions. 60 action limit, 5 collision max.
**SB26**: Tile matching with 64 energy. ACTION5=confirm, ACTION6=click, ACTION7=undo.
**VC33**: Click sprites for pixel swaps/animations. Win: goal sprites match states.
**DC22**: Keyboard+click, 4 kbd + 2 click actions. Deep solutions (64 baseline).

### Games Not Yet Analyzed

AR25, BP35, CN04, KA59, LF52, M0R0, RE86, S5I5, SK48, SU15, WA30

---

## Framework Architecture (For API Submission)

The goal is a pure CPU framework that plays all 25 games via API:

```
┌─────────────────────────────────────┐
│         Game Solver Framework        │
├─────────────────────────────────────┤
│  1. Pre-computed Solutions           │ ← LP85, TN36, FT09, TU93, SP80...
│     (replay saved action sequences)  │
├─────────────────────────────────────┤
│  2. Per-Game Abstract Solvers        │ ← Source-analyzed, game-specific
│     (algebraic, constraint, BFS)     │
├─────────────────────────────────────┤
│  3. Generic BFS Explorer             │ ← UCB1 + frame hash (v6)
│     (fallback for unsolved games)    │
└─────────────────────────────────────┘
```

### Key Design Decisions
- **Reset+replay** instead of deepcopy for correctness
- **Abstract state** instead of frame hash for speed
- **Per-game solvers** are 100-1000x faster than generic BFS
- **Pre-computed solutions** are instant (0ms per level)

---

## Environment Setup

```bash
cd C:\Users\atchi\h4-polytopic-attention
.venv-arc3\Scripts\activate

# Run individual solvers
cd scripts
python solve_lp85.py      # LP85 8/8 in 0.2s
python solve_tn36.py      # TN36 7/7 in 0.4s
python solve_ft09.py      # FT09 3/6

# Run generic BFS on specific games
python solver_fast.py tu93 cd82

# Run all games
python solver_fast.py
```

---

## API Keys
- **ARC-AGI-3:** `58b421be-5980-4ee8-8e57-0f18dc9369f3`
- **Python venv:** `.venv-arc3` (Python 3.12 + arcengine)
- **Game files:** `environment_files/<game_id>/<hash>/<game_id>.py`

---

## Next Session Priorities

1. **Extend FT09** to 6/6 with GF(3) Gaussian elimination
2. **Build G50T abstract solver** — grid movement BFS without engine
3. **Build SC25 solver** — spell pattern enumeration + pathfinding
4. **Analyze remaining 11 games** — read source, plan solvers
5. **Build unified framework** — single entry point for all games
6. **API integration** — connect pre-computed + live solving to API
