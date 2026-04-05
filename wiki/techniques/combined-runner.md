# Combined Runner

## File: `scripts/run_combined.py`

## Architecture
1. Load pre-computed solutions from `solutions/*.json`
2. For each game: replay precomputed actions (instant)
3. If precomputed got 0 levels: fall back to explorer v6 (subprocess, 600s timeout)
4. Take MAX of precomputed vs explorer
5. Report scorecard

## Key Bugs Fixed
- **ACTION1 init**: Runner sent hardcoded ACTION1 to initialize, eating the first solution action. Fixed: use first solution action as init.
- **LP85 explorer hang**: Explorer v6 on LP85 exhausts runner memory. Fixed: LP85 now uses precomputed (5/8).
- **Subprocess timeout**: Thread-based timeout didn't kill the process. Fixed: subprocess with `timeout` parameter.
- **BP35/LF52 timeout**: Explorer hangs on these. Subprocess timeout (600s) kills cleanly.

## Configuration
```python
SKIP_GAMES = set()           # Games to skip entirely
FORCE_EXPLORER = set()       # Games to always use explorer
EXPLORER_BUDGETS = {}        # Per-game budget overrides
```

## Solution Formats Supported
- `{"actions": [{"id": 6, "data": {"x": 20, "y": 14}}]}` — click actions
- `{"actions": [{"id": 1}]}` — keyboard actions
- `{"button_indices": [0, 1, 2]}` — resolved at runtime via `_get_valid_clickable_actions()`

## GitHub Actions
- Workflow: `.github/workflows/arc3-solve.yml`
- Branch: `claude/polytopic-attention-implementation-XHkL3`
- Trigger: Manual dispatch from Actions tab
