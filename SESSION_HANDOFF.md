# Session Handoff — ARC-AGI Self-Compiling Intelligence

**Date:** 2026-03-28 (session 2)
**Status:** ARC-AGI-1 at 395/400 (98.75%), ARC-AGI-3 agent at 4.4%+ and improving

---

## ARC-AGI-1: 395/400 (98.75%)

### Solve Waves
| Wave | Tasks | Solved | Method |
|------|-------|--------|--------|
| Original mechanical | 400 | 38 | C/TVM compiled patterns |
| Agent batches b0-b24 | 262 | 262 | Claude Code parallel agents |
| Agent batches b25-b34 | 99 | 81 | Claude Code parallel agents (10 batches of 10) |
| Retry batches a/b/c | 18 | 12 | Focused retry on hardest tasks |
| Final-6 attempt | 6 | 2 | Third attempt with creative approaches |
| **Total** | **400** | **395** | |

### 5 Remaining Unsolved Tasks
- `234bbc79` — Complex grid compression with 5-markers indicating movement/bending
- `3631a71a` — 30×30 grid with color-9 holes, complex inpainting (not simple symmetry)
- `e40b9e2f` — Body shape + isolated dot → rotational symmetry (center/transform rule unclear)
- `f8c80d96` — Spiral/nested bracket patterns extended to fill grid

### Solution Files
```
data/arc_python_solutions_b{0-34}.json    # Batches 0-34
data/arc_python_solutions_retry_{a,b,c}.json  # Retry waves
data/arc_python_solutions_final6.json     # Final push
olympus/wasm_tools/arc/solved/*.c         # 38 compiled C programs
```

---

## ARC-AGI-3: Self-Compiling Agent

### Architecture: `olympus/arc3/solver.py`
```
Perception (object tracking, centroid analysis)
    → Rule Extraction (movement, walls, goals from frame diffs)
    → Graph Explorer (persistent state graph with BFS to frontier)
    → Click Explorer (per-state click tracking with priority ordering)
    → Cross-Level Memory (bootstrap rules from level N to N+1)
    → Winning Path Replay (instant replay of solved levels on retry)
```

### Current Results: 8/182 levels (4.4%)
| Game | Levels | Notes |
|------|--------|-------|
| ar25 | 2/8 | Cross-level bootstrapping pushed to level 2 |
| cd82 | 1/6 | |
| ft09 | 1/6 | Click game, solved by per-state click tracking |
| lp85 | 1/8 | Click game |
| m0r0 | 1/6 | Winning path replay: 28 states (was 608) |
| r11l | 1/6 | |
| sp80 | 1/6 | |
| tu93 | 1/9 | |

### Key Improvements Over Previous Agent
1. **Click fix**: ACTION6 needs `data=action.action_data.model_dump()` passed explicitly
2. **Per-state click tracking**: Click targets tracked per frame hash, not globally
3. **Persistent graph across retries**: Don't reset state graph on GAME_OVER
4. **Cross-level bootstrapping**: Player color, walls, movement rules transfer to next level
5. **Winning path replay**: Solved levels replayed instantly on retry, saving action budget
6. **Status bar masking**: Edge-hugging regions and always-changing pixels masked

### Scaling Path to 12.58%+
- 3rd place used 91K actions/game over 7.9 hours
- Our 5K run: 4.4% in 6 minutes
- 50K run in progress — expected 40-60 minutes
- Key bottleneck: state space coverage, not agent intelligence

### Reference Repos (cloned locally)
```
ARC-AGI-3-Agents/           # Official agent template
arc-agi-3-just-explore/     # 3rd place solution (graph BFS, 12.58%)
ARC3-solution/              # StochasticGoose (CNN-based, competitive)
```

---

## Environment Setup

```bash
# ARC-AGI-1
py solve_arc_b{N}.py                    # Verify batch N solutions

# ARC-AGI-3
source .venv-arc3/Scripts/activate      # Windows
ARC_API_KEY="58b421be-5980-4ee8-8e57-0f18dc9369f3" py olympus/arc3/solver.py [game_id] [max_actions]

# All games
ARC_API_KEY="..." py olympus/arc3/solver.py  # Runs all 25 games
```

---

## What To Do Next

### Priority 0: Scale ARC-AGI-3
- Run with 91K+ actions per game (match 3rd place budget)
- Optimize per-action compute time (currently ~3ms, 3rd place ~0.3ms)
- The agent architecture is solid — it just needs more exploration budget

### Priority 1: Improve ARC-AGI-3 Exploration
- Integrate 3rd place's priority-group system for click targets (salient color + medium size first)
- Better frontier navigation (precomputed distance maps like 3rd place)
- Handle games with very large state spaces (cn04: 2266 states, still unsolved)

### Priority 2: ARC-AGI-1 Last 4
- 234bbc79, 3631a71a, e40b9e2f, f8c80d96
- These survived 3+ attempts each — may need fundamentally different approaches

### Priority 3: ARC-AGI-3 Level 2+ Breakthrough
- ar25 solves level 2 with bootstrapping — other games should too
- Need to understand why games get stuck after level 1
- Possibly: level 2+ has different mechanics that bootstrapped rules don't cover

---

## Critical Notes
- **DO NOT read game source code** for ARC-AGI-3 (it's the answer key)
- **ACTION6 requires explicit data**: `env.step(action, data=action.action_data.model_dump())`
- **Token limit for TVM**: 10,000,000 max_tokens in verifier.py
- **WASM stack limit**: 4KB. Use `static` arrays in C programs
- **RunPod pods**: All STOPPED, volumes preserved
- **Lattice app** at http://127.0.0.1:7860
