---
game_id: sk48
tags: deep_stuck, grid_fine, high_depth
confidence: 0.8
runs_seen: 4
start_mode: grid_fine
switch_mode_at: 20000
switch_mode_to: grid_fine
---

## Observation
sk48: 187-4.6K states, depth 142. Deep exploration but stuck. 0/8 levels.
Gets deep into the state graph but never reaches a solution.

## Lesson
High depth + zero levels = following a single long path that doesn't terminate.
Needs breadth, not depth. Grid_fine with more diverse actions should help.
The agent is clicking the same thing repeatedly and going deeper.

## Action
Start grid_fine. The segment approach goes too deep on one path.
Grid_fine forces broader exploration across the frame.
