---
game_id: r11l
tags: low_efficacy, grid_fine, efficacy_switch
confidence: 0.8
runs_seen: 2
start_mode: grid_fine
min_efficacy: 0.05
switch_mode_to: grid_fine
---

## Observation
r11l: segment mode has near-zero efficacy. Efficacy-based switching (v6) caught this
and switched to grid_fine, solving level 1. Before v6: 0/6. After: 1/6.

## Lesson
Some games have no clickable segments that matter. Segment clicks do nothing.
Grid-fine mode (step=2) is the right approach from the start.

## Action
Start in grid_fine mode. Don't waste actions on segment clicking.
