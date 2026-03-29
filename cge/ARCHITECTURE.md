# CGE Architecture: Deep Technical Documentation

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Core Algorithm](#core-algorithm)
3. [Component Deep Dives](#components)
4. [The Breakthrough Agent](#breakthrough-agent)
5. [MCTS Sub-Agent](#mcts-sub-agent)
6. [Smart Replay](#smart-replay)
7. [Environment Design](#environments)
8. [Failure Modes and Fixes](#failures)
9. [Reproducing Results](#reproducing)
10. [Future Work](#future-work)

---

## Problem Statement <a name="problem-statement"></a>

Given an interactive environment with:
- **States** (identified by string hashes)
- **Actions** (integer IDs, some change state, some don't)
- **Episodes** (budget of N actions, then GAME_OVER reset)
- **Levels** (solve one to advance, must replay solved levels after reset)

Find the action sequence that solves all levels using minimum total actions.

**Why BFS fails**: BFS explores every action at every state equally. In a tree
with 8 branches and 1 correct path, BFS wastes 7/8 of its budget. Over 4 depth
levels: (7/8)^4 = 59% wasted. With tight episode budgets, BFS can't go deep
enough to find solutions.

**Why random fails**: Random exploration finds states but can't navigate back to
them. Without memory, it rediscovers the same dead ends every episode.

**What we need**: An algorithm that (1) remembers what it's tried, (2) learns
which actions are productive, and (3) concentrates budget on promising branches.

---

## Core Algorithm <a name="core-algorithm"></a>

### The Explore-Compress-Synthesize Loop

```
while not solved:
    EXPLORE:   Try actions, build state transition graph
    COMPRESS:  Learn patterns from graph (action efficacy, state types, dead ends)
    SYNTHESIZE: Use patterns to guide future exploration (UCB1, compression ranking)
    REPEAT
```

### Action Selection: Multi-Armed Bandits

Each (state, action) pair is a bandit arm. The reward is the number of NEW states
discovered downstream. UCB1 selects which arm to pull:

```
score(state, action) = mean_reward + C_eff * sqrt(ln(N) / n_i)

where:
  mean_reward = average new states discovered when taking this action at this state
  N           = total visits to this state across all episodes
  n_i         = visits to this specific (state, action) pair
  C_eff       = C / (1 + n_i / 10)  — decays as confidence grows
```

The adaptive C is critical: it starts high (explore all options) and decays
(exploit known-good options). After 10+ visits, C_eff ≈ C/2. After 100+ visits,
C_eff ≈ C/11. This naturally transitions from exploration to exploitation.

### Dead Action Pruning

Actions with `n_i >= 10 AND mean_reward == 0` get score `-1.0`, permanently
removing them from consideration. This prevents wasting budget on known-useless
actions (e.g., actions 4-7 in NeedleInHaystack that never change state).

### Discriminative Signal Detection

UCB1 only overrides random selection when it has meaningful data. The test:

```python
def _has_discriminative_signal(state):
    rewards = [mean_reward for each tested action at state]
    return max(rewards) > mean(rewards) * 2  # best is 2x average
        or max(rewards) - min(rewards) > 0.5  # clear gap
```

When UCB has no signal (new state, few visits), falls back to CompressionLayer's
state-dependent ranking (proven on CausalChain) or random selection.

---

## Component Deep Dives <a name="components"></a>

### core.py — GraphExplorer

The state graph tracks nodes (states), edges (transitions), and frontier
(nodes with untested actions). Navigation uses reverse BFS from frontier,
with depth-biased tie-breaking (prefer deeper frontier nodes).

Key methods:
- `add_node(state, actions)`: Register a new state with its available actions
- `record_transition(src, action, changed, target)`: Record action result
- `choose_action(current, action_order)`: Pick next action using UCB or navigation

The GraphExplorer assumes **bidirectional navigation**: if there's an edge
A→B via action X, it creates a reverse edge B→A and routes through it.
**This assumption fails in tree environments** where you can't go backwards.
This is why the MCTS sub-agent exists.

### compression.py — CompressionLayer

Periodically scans the graph and learns:
- **Action efficacy**: per-action success rate (global and state-dependent)
- **State signatures**: (change_rate, fanout, depth, total_tested)
- **Bottlenecks**: states with exactly one novel successor
- **Progress direction**: average signature trajectory along winning paths

The `rank_actions(state, available)` method returns actions ordered by predicted
quality, using state-dependent stats when available (3+ attempts at that state
type), falling back to global stats.

State classification: extracts features from state name strings (parts, prefixes).
In a real ARC system, this would use frame features (colors, shapes, positions).

### agent_breakthrough.py — BreakthroughAgent + SmartExplorer + _MCTSSub

Three classes in one file:

1. **SmartExplorer**: GraphExplorer subclass with UCB1, adaptive C, dead action
   pruning, per-state reward tracking, and feature-based learning infrastructure.

2. **BreakthroughAgent**: The main agent that orchestrates SmartExplorer,
   CompressionLayer, MCTS sub-agent, replay, and stall detection.

3. **_MCTSSub**: Completely independent MCTS sub-agent with its own state
   tracking, descendant credit, and forced rotation. Shares NOTHING with
   the SmartExplorer.

---

## The Breakthrough Agent <a name="breakthrough-agent"></a>

### Initialization

```python
agent = BreakthroughAgent(
    analyze_interval=150,      # run compression analysis every N actions
    learn_threshold=50,        # start using compression after N actions
    exploration_constant=1.5,  # UCB1 C parameter
)
```

### Main Loop (choose_action)

```
1. If replaying → check if replay needed (smart replay), return replay action
2. If MCTS sub active → delegate entirely to _MCTSSub.choose()
3. Normal mode:
   a. Check if UCB has discriminative signal at current state
   b. If yes → UCB1 select among untested actions
   c. If no → use compression ranking or random
   d. If all actions tested → navigate toward frontier via BFS
```

### Observe Result

```
1. If MCTS sub active AND not replaying → delegate to _MCTSSub.observe()
2. Normal mode:
   a. Record transition in explorer
   b. Track action efficacy
   c. Track global change rates (for adaptive pruning)
```

### Episode Reset

```
1. If MCTS sub active → delegate reset, rebuild replay queue
2. Normal mode:
   a. Count explorer states — if no growth, increment stall counter
   b. If stall >= 10 → spawn _MCTSSub
   c. Finalize pending UCB rewards
   d. Rebuild replay queue from winning paths
```

### Level Complete

```
1. Save winning path (effective_history, not full history)
2. Update transfer bias (winning action types persist)
3. Tell compression about the win
4. Reset explorer for new level
5. If MCTS sub active → reset level-specific state tracking
```

---

## MCTS Sub-Agent <a name="mcts-sub-agent"></a>

### Why it exists

The GraphExplorer assumes bidirectional navigation. In tree environments
(DeepTreeSearch), edges are one-way. The explorer records reverse edges
but they don't correspond to real environment transitions. When the explorer
tries to navigate "back" to root via reverse edges, the environment goes
somewhere else entirely. The explorer gets stuck cycling through the same
few states.

The MCTS sub-agent doesn't use the GraphExplorer at all. It makes decisions
purely from cumulative descendant data collected across episodes.

### How it works

```python
class _MCTSSub:
    def choose(state, available_actions):
        # 1. Forced rotation: try each action at each state at least once
        min_tries = min(sa_tries[(state, a)] for a in actions)
        candidates = [a for a if sa_tries[(state, a)] <= min_tries]

        # 2. Exploit: if all tried and clear winner, focus on it
        if min_tries > 0:
            best_desc = max(sa_descendants[(state, a)] for a in actions)
            if best_desc > 3:
                candidates = [a if sa_descendants[(state, a)] >= best_desc * 0.5]

        return random.choice(candidates)

    def observe(prev_state, action, new_state, changed):
        if changed and new_state not in level_states:
            level_states.add(new_state)
            # Credit ALL ancestors in episode path
            for (s, a) in episode_path:
                sa_descendants[(s, a)] += 1
```

### Key design decisions

1. **Forced rotation before exploitation**: Every action gets tried at least once
   at every state. This prevents early commitment to a bad branch based on
   insufficient data.

2. **Cumulative descendants (not per-episode)**: `sa_descendants` accumulates
   across ALL episodes. Branch 2 with 56 descendants across 10 episodes scores
   higher than branch 0 with 3 descendants. The signal gets stronger over time.

3. **Ancestor credit**: When a new state is found 4 levels deep, ALL 4 ancestors
   in the episode path get credited. This propagates deep signal to the root:
   "root action 2 leads to the most discoveries overall."

4. **Level-scoped state tracking**: `_level_states` tracks states discovered in
   the CURRENT level only. `all_states` tracks globally. New-state detection
   uses `_level_states` to avoid counting previously-seen (replayed) states.

5. **Independence**: Shares NOTHING with the SmartExplorer. No nodes, no edges,
   no tested dict, no frontier. This is deliberate — all integration attempts
   that shared state failed due to the explorer's bidirectional assumption.

---

## Smart Replay <a name="smart-replay"></a>

### The problem

After GAME_OVER, the agent replays solved levels' winning paths to return to the
current unsolved level. This works for environments where GAME_OVER resets to
level 0 (like ARC-AGI-3). But some environments reset to the current unsolved
level directly (like DeepTreeSearch).

When replay runs on an environment that doesn't need it, the replay actions
are meaningless at the current level's states. Action 0 at level 1's root
goes somewhere different than action 0 at level 0's root. The replay poisons
exploration AND triggers on_level_complete which resets accumulated MCTS data.

### The fix

```python
if self.replaying and self.replay_queue:
    # Check: does the current state match a solved level's root?
    if state not in self._replay_root_states():
        # Environment already reset to current level — skip replay
        self.replaying = False
        self.replay_queue = []
    else:
        # Environment reset to level 0 — replay needed
        action = self.replay_queue.pop(0)
        return action
```

`_replay_root_states()` returns the starting states of all solved levels.
If the current state doesn't match any of them, the environment already
jumped to the current level and replay is unnecessary.

### Impact

DeepTreeSearch: 20/80 → 47/80 (+135%). This single fix was the difference
between "stuck" and "breakthrough" because it gave the MCTS sub full
60-action episodes instead of 53-action episodes polluted by stale replay.

---

## Environment Design <a name="environments"></a>

### Design philosophy

Each environment tests a specific failure mode of BFS:

| Environment | BFS failure mode | What CGE learns |
|------------|-----------------|-----------------|
| LinearPuzzle | Wastes actions on wrong action types | Which actions produce results |
| MazeNavigation | Tries useless click actions | Arrows >> clicks |
| BottleneckPuzzle | Explores dead branches equally | Branch depth predicts productivity |
| HiddenPatternPuzzle | Can't learn state→action mapping | Feature-dependent action selection |
| LargeStateSpace | Spreads budget too thin | Focus on productive corridors |
| DeepTreeSearch | Can't backtrack in trees | MCTS forced rotation + descendant credit |
| NeedleInHaystack | Wastes budget on useless actions | Dead action pruning (adaptive C) |
| StuckGame | Most actions do nothing | Quickly identify the 2 working actions |
| CausalChain | Single actions don't help | Compression learns action sequences |
| RuleLearning | Can't transfer rules across levels | Cross-level transfer bias |

### Environment interface

```python
class Environment:
    def reset(self) -> (state: str, available_actions: set)
    def step(action) -> (state, available_actions, changed: bool, level_up: bool, game_over: bool)
```

All environments are deterministic given a random seed. The benchmark uses
fair seeding: same seed for environment creation, same seed for agent
randomness, across all agents being compared.

---

## Failure Modes and Fixes <a name="failures"></a>

### 1. Transfer bias locks onto wrong action (DeepTreeSearch)

**Symptom**: Agent always picks action 0 at level 1 because level 0's
winning path used action 0.

**Root cause**: `_build_action_order` puts transfer-biased actions first.
Explorer uses action_order for untested action selection, always picking
the first matching action.

**Status**: Partially mitigated by MCTS sub (which ignores transfer bias).
Full fix would require transfer at the FEATURE level (parity, not action ID).

### 2. GraphExplorer can't navigate trees (DeepTreeSearch)

**Symptom**: Explorer records reverse edges that don't work. Gets stuck
cycling through 4 states for hundreds of episodes.

**Root cause**: Explorer assumes if A→B exists, then B→A is navigable.
In trees, you can only go forward.

**Fix**: MCTS sub-agent with no shared state (Layer 3).

### 3. Replay poisons tree exploration (DeepTreeSearch)

**Symptom**: MCTS sub finds 65-76 states but never reaches GOAL.

**Root cause**: Every episode starts with 7 useless replay actions.
`on_level_complete` during replay resets MCTS data.

**Fix**: Smart replay detection (Layer 4). Skip replay when environment
resets to current level.

### 4. Dead-end prediction causes false avoidance (v2)

**Symptom**: Agent avoids states that look like dead ends but aren't.
Regresses on BottleneckPuzzle.

**Root cause**: Feature-based dead-end prediction generalizes from few
examples. Features like "depth_range" match both dead and live states.

**Fix**: Removed dead-end prediction from production agent. Left infrastructure
in compression_v2.py for future research with more robust classifiers.

### 5. Stall threshold triggers on slow environments (CausalChain)

**Symptom**: MCTS sub activates on CausalChain where it shouldn't.

**Root cause**: CausalChain has naturally slow state discovery (most actions
don't change state). Stall threshold of 5 falsely triggers.

**Fix**: Raised threshold to 10. CausalChain never has 10 consecutive
zero-growth episodes.

---

## Reproducing Results <a name="reproducing"></a>

### Exact reproduction

```bash
python3 -m cge.benchmark
```

This runs 20 seeds × 10 environments × 3 agents (BFS, Best, UCB) with
fair seeding. Expected output:

```
TOTAL    BFS ~390    Best ~456    UCB ~492
```

Variance across runs is ±5 levels due to environment randomization.

### Single environment testing

```python
import random
from cge.environments_hard import DeepTreeSearch
from cge.agent_breakthrough import BreakthroughAgent

random.seed(42)
env = DeepTreeSearch(n_branches=8, depth=4, n_levels=4, episode_budget=60)
agent = BreakthroughAgent()

state, actions = env.reset()
agent.on_new_state(state, actions)
while True:
    action = agent.choose_action(state, actions)
    if action is None:
        action = random.choice(list(actions))
    prev = state
    state, actions, changed, level_up, game_over = env.step(action)
    agent.observe_result(prev, action, state, actions, changed)
    if level_up:
        agent.on_level_complete(env.current_level - 1)
    if game_over:
        if env.current_level >= env.total_levels:
            break
        agent.on_episode_reset()
        state, actions = env.reset()
        agent.on_new_state(state, actions)

print(f"Solved {env.current_level}/{env.total_levels} levels")
```

### Adding a new environment

1. Add class to `environments_hard.py` implementing `reset()` and `step()`
2. Add to `get_hard_environments()` factory function
3. Run `python3 -m cge.benchmark` to see how all agents perform
4. If UCB loses, diagnose: is it UCB signal, compression, stall detection, or replay?

### Adding a new agent

1. Create `agent_newname.py` implementing the agent interface:
   - `on_new_state(state, actions)`
   - `choose_action(state, actions) → action`
   - `observe_result(prev_state, action, new_state, new_actions, changed)`
   - `on_level_complete(level)`
   - `on_episode_reset()`
2. Add to `benchmark.py` agent_types dict
3. Run benchmark

---

## Future Work <a name="future-work"></a>

### Near-term: ARC-AGI-3 integration

Replace action selection in `olympus/arc3/explorer_v4.py` with per-state UCB1.
The frame hash is the state, the segment/grid click is the action. Reward =
new frame hashes discovered. Expected improvement: 13-15% based on simulated
environment results.

### Medium-term: Feature-based transfer

Learn "even-indexed actions are productive" as an abstract rule that transfers
across states and levels. Infrastructure exists (`_action_features` in
SmartExplorer) but integration failed due to transfer bias overriding
untested action exploration. Needs: feature-based UCB priors that ORDER
untested actions without BLOCKING exploration of other actions.

### Long-term: The DeepTreeSearch ceiling

Current: 47/80 (59%). Standalone MCTS: 150/200 (75%). The 16% gap is from:
1. Budget spent on level 0 exploration before MCTS triggers (~3000 actions)
2. 10-episode stall detection delay
3. MCTS rotation using random.choice instead of intelligent ordering

Potential fixes:
- Start in MCTS mode if first episode reveals tree structure (all actions
  from root lead to unique children, no reverse transitions)
- Feature-based action ordering during rotation (even-index first)
- Reduce stall threshold for environments with clear tree signatures
