---
game_id: ft09
tags: deep_exploration, productive, dont_switch
confidence: 1.0
runs_seen: 3
---

## Observation
ft09 explores 167K states at depth 184. This is PRODUCTIVE deep exploration —
it solves 2/6 levels. Previous stall-triggered mode switching killed this game,
dropping from 2/6 to 0/6 in one run.

## Lesson
Not all deep exploration is stuck. ft09 needs depth to solve.
Never trigger mode switch on ft09 based on stall detection.
High state count + solving levels = leave it alone.

## Action
Keep segment mode for ft09. Budget 350K. Do NOT switch modes based on stall count.
Only switch if efficacy is truly zero (which it isn't for ft09).
