---
game_id: cn04
tags: efficacy_switch, segment_then_grid
confidence: 0.7
runs_seen: 1
switch_mode_at: 15000
switch_mode_to: grid
min_efficacy: 0.05
---

## Observation
cn04: segment mode explored 51K states but efficacy was low. Efficacy switch
caught it and moved to grid. Solved 1/5. New in v6.

## Lesson
cn04 benefits from segment exploration first (builds state graph) then grid mode.
Switch earlier than default — 15K actions instead of 20K.

## Action
Start segment, switch to grid at 15K if efficacy <5%.
