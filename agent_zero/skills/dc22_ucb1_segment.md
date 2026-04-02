---
game_id: dc22
tags: segment, ucb1, medium_states
confidence: 1.0
runs_seen: 3
boost_game_actions: 6
---

## Observation
dc22 consistently gains +1 level when UCB1 is active within segment mode.
v4 (no UCB1): 2/6. v5+ (UCB1): 3/6. Reproducible across 3 runs.
1123 states explored, 300K budget.

## Lesson
UCB1 within segment mode finds productive click targets that BFS misses.
Segment mode works well — don't switch to grid.

## Action
Keep segment mode. Let UCB1 run. Budget 300K+. Click actions (game_action 6) are productive.
