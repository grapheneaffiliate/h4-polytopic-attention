---
game_id: sb26
tags: grid_fine, shallow, low_states
confidence: 0.7
runs_seen: 4
start_mode: grid_fine
budget_override: 300000
---

## Observation
sb26: 313-8K states. Grid-fine, shallow exploration. 0/8 levels.
Similar profile to wa30 — grid is right but not cracking it.

## Lesson
Shallow + grid-fine + zero levels = clicking in the wrong places.
Need smarter grid targeting — click on edges, colored pixels, not background.

## Action
Start grid_fine. Budget 300K. Arrow keys may also be important.
