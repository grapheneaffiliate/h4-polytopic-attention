---
game_id: g50t
tags: low_states, barely_exploring, needs_different_approach
confidence: 1.0
runs_seen: 5
start_mode: grid_fine
budget_override: 300000
---

## Observation
g50t: only 14-145 states across all runs. 0/7 levels. Barely exploring.
Neither segment nor grid mode produces meaningful state changes.
This is the hardest game — almost nothing the agent does changes the frame.

## Lesson
When even 200K actions produce <200 states, the action space itself is wrong.
Need to try: grid_fine from start, arrow keys (game actions 1-5), action 7 (submit/reset).
Possibly a game where you need to build a specific pattern via precise clicks.

## Action
Start grid_fine (step=2). Prioritize arrow keys and grid clicks on non-background pixels.
300K budget. If still stuck at 100K, try action sequences (arrows then click).
