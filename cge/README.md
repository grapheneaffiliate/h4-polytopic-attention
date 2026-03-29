# CGE: Compression-Guided Exploration

**A self-learning search algorithm that outperforms BFS by 26% on diverse environments.**

Final benchmark (20 seeds, 10 environments, 10K action budget):
```
UCB Breakthrough:  492 levels, 27,870 act/run  (+26% vs BFS)
CGE Best:          456 levels, 31,365 act/run  (+17% vs BFS)
BFS baseline:      390 levels, 36,581 act/run
```

## Quick Start

```bash
# From the repo root:
python3 cge/tests/test_core.py          # 10 core graph tests
python3 cge/tests/test_compression.py   # 8 compression tests
python3 cge/tests/test_agent.py         # 8 agent tests
python3 cge/tests/test_environments.py  # 10 environment tests

python3 -m cge.benchmark               # full 20-seed, 10-env benchmark (~5 min)
python3 -m cge.demo                     # quick comparison demo
```

Requires only Python 3.10+ and the standard library. No external dependencies.

---

## The Algorithm

### One-sentence summary

Treat action selection as a **multi-armed bandit** with per-state rewards, use **compression** as fallback when bandit signal is weak, and spawn an **independent MCTS sub-agent** when the explorer gets stuck in tree-like environments.

### The four layers

```
Layer 1: Per-State UCB1 (primary action selection)
  At each state, track (state, action) → new states discovered.
  UCB1 formula with adaptive C that decays as confidence grows.
  Actions with 0 reward after 10+ visits get permanently pruned.

Layer 2: Compression Fallback (when UCB has no signal)
  CompressionLayer analyzes the graph periodically.
  Provides state-dependent action ranking.
  Key for environments like CausalChain where all actions have similar
  per-action reward but state features predict the correct action.

Layer 3: MCTS Sub-Agent (when explorer is stuck)
  Stall detection: 10+ consecutive episodes with 0 new states.
  Spawns _MCTSSub — completely independent, no shared state with explorer.
  Forced rotation: tries each action at each state at least once.
  Descendant credit: when a new state is found, ALL ancestors in the
  episode path get credit. Concentrates budget on productive branches.

Layer 4: Smart Replay (handles different reset behaviors)
  Some environments reset to level 0 on GAME_OVER (need replay of solved levels).
  Others reset to current unsolved level (replay would poison exploration).
  Detects which by checking if reset state matches a solved level's root.
```

### Why it works

BFS explores everything equally — 7/8 of its budget is wasted on dead-end branches in a tree with 8 branches and 1 solution. UCB1 learns from experience and concentrates budget on productive branches. The adaptive C ensures it explores enough to discover the right branch but then exploits it aggressively.

The key insight: **actions are bandits, not random choices**. Each (state, action) pair has a reward history (new states discovered). UCB1 balances exploration of untried actions vs exploitation of known-good ones.

---

## File Structure

```
cge/
├── README.md                  # This file
├── ARCHITECTURE.md            # Deep technical documentation
├── __init__.py                # Package exports
│
├── core.py                    # GraphExplorer: state graph with BFS navigation
├── compression.py             # CompressionLayer v1: state-dependent action ranking
├── compression_v2.py          # CompressionLayer v2: dead-end/rules/sequences (experimental)
│
├── agent.py                   # CGEAgent (v1) + BFSAgent baseline
├── agent_v2.py                # v2: compression v2 integration (experimental)
├── agent_v3.py                # v3: branch pruning + transfer (experimental)
├── agent_best.py              # Best: v1 compression + v3 transfer + pruning
├── agent_breakthrough.py      # UCB Breakthrough: THE winning agent (492 levels)
│
├── environments.py            # 5 standard test environments
├── environments_hard.py       # 5 hard environments (modeled on ARC-AGI-3)
│
├── benchmark.py               # Fair-seeded 20-seed benchmark
├── demo.py                    # Quick comparison demo
│
└── tests/
    ├── __init__.py
    ├── test_core.py           # 10 tests for GraphExplorer
    ├── test_compression.py    # 8 tests for CompressionLayer
    ├── test_agent.py          # 8 tests for CGEAgent + BFSAgent
    └── test_environments.py   # 10 tests for all environments
```

### Which files matter

If you're **using** this: `agent_breakthrough.py` is the winning agent. Run `benchmark.py`.

If you're **understanding** this: read `core.py` → `compression.py` → `agent.py` → `agent_breakthrough.py` in that order.

If you're **extending** this: add new environments to `environments_hard.py`, new agents alongside existing ones, test with `benchmark.py`.

---

## Environments

### Standard (environments.py)

| Name | What it tests | Properties |
|------|--------------|------------|
| LinearPuzzle | Action efficacy learning | 5 correct actions out of 15, tight budget |
| MazeNavigation | Spatial exploration | 4 useful arrows out of 10 actions |
| BottleneckPuzzle | Dead-end avoidance | 5 branches, 1 correct, deep dead ends |
| HiddenPatternPuzzle | State→action mapping | State features determine correct action |
| LargeStateSpace | Budget focusing | 100 states, 2 useless actions |

### Hard (environments_hard.py)

| Name | Models | Properties |
|------|--------|------------|
| DeepTreeSearch | Tree environments | 8 branches × 4 depths, 60-action episodes, one-way |
| NeedleInHaystack | re86 (5807 states) | 15×15 grid, waypoints, 4/8 actions useful |
| StuckGame | sk48 (5 states) | 10/12 actions useless, must find the 2 that work |
| CausalChain | Multi-step patterns | Must discover 3-action sequences |
| RuleLearning | Feature→action rules | State features predict correct action, transfers across levels |

---

## Benchmark Results

### Final (20 seeds, fair-seeded)

```
Environment               BFS lvl   Best lvl    UCB lvl     UCB advantage
─────────────────────────────────────────────────────────────────────────
LinearPuzzle                60/80      60/80      60/80      —
MazeNavigation              40/80      40/80      40/80      1.6x faster
BottleneckPuzzle            29/80      25/80      25/80      BFS wins here
HiddenPatternPuzzle         60/80      60/80      60/80      1.3x faster
LargeStateSpace             20/80      20/80      20/80      2.1x faster
DeepTreeSearch              20/80      20/80      47/80      2.35x more levels
NeedleInHaystack            21/80      31/80      40/80      1.9x more levels
StuckGame                   40/80      40/80      40/80      —
CausalChain                  0/80      60/80      60/80      1.4x faster
RuleLearning               100/80     100/80     100/80      1.1x faster
─────────────────────────────────────────────────────────────────────────
TOTAL                       390        456        492        +26% vs BFS
```

### Evolution (how we got here)

```
BFS           → 390 levels  (baseline: explores everything equally)
CGE v1        → 444 levels  (action efficacy + state-dependent ranking)
CGE v2        → 442 levels  (dead-end/rules too noisy — REGRESSED)
CGE v3        → 428 levels  (branch pruning great for grids, kills CausalChain)
CGE Best      → 456 levels  (v1 compression + v3 transfer + adaptive pruning)
UCB           → 445 levels  (per-state UCB1 + adaptive C)
UCB + MCTS    → 463 levels  (dual-mode with stall detection)
UCB + replay  → 492 levels  (smart replay skip for tree environments)
```

---

## Integration with ARC-AGI-3

The CGE framework is designed to drop into the ARC-AGI-3 explorer. The key integration points:

### In `olympus/arc3/explorer_v4.py`:

1. **Replace `random.choice(untested)`** with `ucb1_select(untested, per_state_rewards)`
2. **Add CompressionLayer** analysis every 150 actions
3. **Add stall detection** to spawn MCTS sub when explorer is stuck
4. **State identity** = frame hash (already computed by the explorer)
5. **Action identity** = (action_type, x, y) tuple (segment click or arrow)
6. **Reward signal** = number of new frame hashes discovered after taking an action

### What stays the same:
- Frame hashing, segmentation, status bar detection
- Priority groups for initial action classification
- Winning path replay across levels
- Episode reset handling

### What changes:
- Action SELECTION within a state (UCB1 instead of random)
- Frontier PRIORITIZATION (score by descendant value, not just depth)
- STALL RECOVERY (MCTS sub instead of cycling through same states)

---

## Key Lessons Learned

### What works
1. **Per-state UCB1** is the single biggest improvement. Global action stats are too noisy.
2. **Adaptive C** (decaying exploration) prevents wasting budget on known-useless actions.
3. **Dead action pruning** (-1 score for 0-reward after 10 visits) is critical for environments with many useless actions (NeedleInHaystack: 4/8 useless).
4. **Compression as fallback** (not primary) — only use when UCB has no discriminative signal.
5. **Independent MCTS sub-agent** — must share NOTHING with the explorer. All integration attempts that shared state failed.
6. **Smart replay skip** — critical for environments that don't reset to level 0.

### What doesn't work
1. **Dead-end prediction from features** — too noisy, causes false positives that avoid productive branches.
2. **Feature-action rule mining** — needs much more data than available (8+ attempts per rule).
3. **Global action stats as UCB fallback** — carries level-specific bias that poisons subsequent levels.
4. **Transfer bias overriding untested actions** — locks agent onto level 0's winning action.
5. **MCTS integrated into explorer** — the GraphExplorer's bidirectional navigation assumption is fundamentally incompatible with tree environments.
6. **Deterministic rotation** (`cands[0]` instead of `random.choice`) — breaks environments where action index order doesn't match quality order.

### Subtle bugs found and fixed
1. **Replay poisoning**: replaying level 0's actions at level 1's root when the environment resets to current level (not level 0). Fix: detect reset target and skip replay.
2. **MCTS sub reset on replay level-up**: `on_level_complete` during replay destroyed accumulated MCTS data. Fix: only reset for genuinely NEW levels.
3. **Level 0 states polluting MCTS level tracking**: `_all_states_ever` included states from previous levels. Fix: separate `_level_states` for per-level tracking.
4. **Stall threshold too aggressive**: threshold=5 triggered on CausalChain (slow but non-zero state discovery). Fix: raised to 10.
5. **`_sa_episode_tries` not incrementing**: MCTS rotation at root only ran once per episode. Fix: forced full episode path tracking.
