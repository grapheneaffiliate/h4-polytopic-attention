---
game_id: lp85
tags: high_performer, randomization, needs_retries
confidence: 1.0
runs_seen: 5
budget_override: 400000
boost_game_actions: 6
---

## Observation
lp85: best performer. 5/8 at 200K, fluctuates 4-5/8 due to randomization.
14840 states. Segment mode works well. 400K budget.

## Lesson
lp85 benefits from more budget and more retries. The randomization in UCB1
means some runs find 5/8 and others find 4/8. More actions = more chances.

## Action
Budget 400K. Stay in segment mode. Click actions are productive.
If we could retry 3x and take best, would stabilize at 5/8.
