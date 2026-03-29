# Compression-Guided Exploration (CGE)

**The universal algorithm: Explore → Compress → Synthesize → Execute → Repeat**

A self-compiling explorer that *learns from its own exploration* to guide future search.
Unlike BFS (which treats every state as equally unknown), CGE discovers patterns —
action efficacy, state types, bottlenecks, progress gradients — and uses them to
exponentially prune the search space.

## Quick Start

```bash
cd cge/
python -m pytest tests/ -v          # run all tests
python tests/test_environments.py   # see the simulated environments
python demo.py                      # full demo: CGE vs BFS on all environments
```

## Architecture

```
cge/
├── core.py              # GraphExplorer (state graph with transitions)
├── compression.py       # CompressionLayer (learns patterns from graph)
├── agent.py             # CGEAgent (the unified explore-compress-synthesize loop)
├── environments.py      # Simulated test environments (no ARC SDK needed)
├── demo.py              # Run CGE vs BFS comparison
└── tests/
    ├── test_core.py
    ├── test_compression.py
    ├── test_agent.py
    └── test_environments.py
```

## Key Ideas

1. **Action Efficacy Learning** — After N actions, the agent knows "arrows work 90%
   of the time, clicks work 10%." Priority groups become *discovered*, not prescribed.

2. **State Signatures** — Each state gets a fingerprint: (change_rate, fanout, depth).
   States with similar signatures are the same *type*. A 5000-state graph might have
   only 4 state types.

3. **Bottleneck Detection** — States with exactly one novel successor are "key moments."
   The winning path goes through bottlenecks. Focus budget past them, not before.

4. **Progress Gradients** — When level 1 is solved, the winning path defines a direction.
   Level 2 probably needs a similar direction. Search toward it instead of BFS everywhere.

5. **Cross-Level Transfer** — Winning path signatures from solved levels guide exploration
   on new levels, avoiding re-exploration of already-understood patterns.

## Relationship to ARC-AGI-3

This is a standalone research prototype. When the algorithm is validated here on
simulated environments, it can be integrated into `olympus/arc3/explorer_v4.py`
for the real ARC competition. The v4 explorer remains untouched.
