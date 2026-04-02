---
game_id: global
tags: low_efficacy, mode_switch
confidence: 0.9
runs_seen: 6
min_efficacy: 0.03
switch_mode_to: grid
---

## Observation
Across all runs: when segment efficacy drops below 3% after 10K+ actions,
the game almost never solves in segment mode. Every game that was rescued
(r11l, tu93, cn04) was rescued by switching to grid or grid_fine.

## Lesson
Low segment efficacy = segment clicks don't interact with the game meaningfully.
Switch faster. The 20K threshold in v6 wastes 10K+ actions on dead clicks.

## Action
If segment efficacy <3% after 10K actions, switch to grid immediately.
