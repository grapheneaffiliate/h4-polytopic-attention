# Agent Zero

**One agent that can do anything. Search when search works, reason when reasoning is needed, learn which to use from experience.**

```
Benchmark (20 seeds, 14 environments, 10K action budget):
  Agent Zero:  748 levels  — BEST
  CGE:         736 levels
  BFS:         682 levels
```

## Quick Start

```bash
python3 -m agent_zero.benchmark       # full benchmark (~8 min)
python3 agent_zero/tests/test_all.py  # 12 tests
```

Zero external dependencies. Python 3.10+ stdlib only.

## Architecture

```
choose_action(state, actions):
  1. Replay solved levels if needed (smart: skip when env resets to current level)
  2. If tree detected → MCTS sub-agent (independent, no shared state)
  3. If stuck → ask Reasoner for action suggestions (soft UCB prior)
  4. UCB1 select among untested (with adaptive C, dead action pruning, reasoner prior)
  5. Navigate toward frontier via BFS (with depth bias)
```

UCB1 is ALWAYS the outer loop. Reasoning provides priors, not overrides.

## Files

```
agent_zero/
├── core.py            # AgentZero: unified UCB1 + MCTS + Reasoner
├── env_interface.py   # Generic Env interface (reset/step)
├── environments.py    # 14 environments (10 from CGE + 4 new)
├── baseline.py        # BFS baseline for comparison
├── reasoner.py        # HeuristicReasoner + LLMReasoner (pluggable)
├── benchmark.py       # Fair-seeded benchmark
└── tests/test_all.py  # 12 tests
```

## Environments

| # | Name | Tests | From |
|---|------|-------|------|
| 1 | LinearPuzzle | Action efficacy | CGE |
| 2 | MazeNavigation | Spatial exploration | CGE |
| 3 | BottleneckPuzzle | Dead-end avoidance | CGE |
| 4 | HiddenPatternPuzzle | Feature→action | CGE |
| 5 | LargeStateSpace | Budget focusing | CGE |
| 6 | DeepTreeSearch | Tree MCTS | CGE |
| 7 | NeedleInHaystack | Dead action pruning | CGE |
| 8 | StuckGame | Find rare actions | CGE |
| 9 | CausalChain | Action sequences | CGE |
| 10 | RuleLearning | Cross-level transfer | CGE |
| 11 | **TextReasoning** | Pattern inference | **NEW** |
| 12 | **CodeEnv** | Incremental editing | **NEW** |
| 13 | **FeatureTransfer** | Generalization (shuffled actions) | **NEW** |
| 14 | **CompositeEnv** | Multi-env planning | **NEW** |

## Plugging in an LLM

```python
from agent_zero import AgentZero
from agent_zero.reasoner import LLMReasoner

# Any callable(str) -> str works
reasoner = LLMReasoner(lambda prompt: my_api_call(prompt))
agent = AgentZero(reasoner=reasoner)
```

The LLM receives a text summary of exploration state and returns
`action_id: weight` lines. Weights become soft UCB priors (Bayesian
update, not hard override).

## How it relates to CGE

Agent Zero is built ON TOP of CGE's proven components:
- Per-state UCB1 with adaptive C (from `cge/agent_breakthrough.py`)
- MCTS sub-agent with descendant credit (from `cge/agent_breakthrough.py`)
- Smart replay skip (from `cge/agent_breakthrough.py`)
- Tree detection (from `cge/agent_breakthrough.py`)

What Agent Zero adds:
- Generic `Env` interface (any environment, not just CGE's)
- Pluggable Reasoner (heuristic or LLM-backed)
- 4 new environment types (text, code, feature transfer, composite)
- Unified agent class (no separate v1/v2/v3/best/breakthrough lineage)
