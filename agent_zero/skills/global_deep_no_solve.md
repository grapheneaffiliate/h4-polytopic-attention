---
game_id: global
tags: deep_stuck, high_depth, no_levels
confidence: 0.7
runs_seen: 4
switch_mode_at: 50000
switch_mode_to: grid_fine
---

## Observation
When depth exceeds 100 and zero levels are solved after 50K actions, the agent
is following one long path that doesn't terminate. Seen in sk48 and re86.

## Lesson
Very high depth + zero progress = the agent found a state chain but not a solution chain.
Switching to grid_fine forces broader exploration instead of deeper.

## Action
If depth >100 and 0 levels at 50K actions, switch to grid_fine regardless of current mode.
