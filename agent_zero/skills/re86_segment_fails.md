---
game_id: re86
tags: high_states_zero_levels, segment_fails, needs_memory
confidence: 1.0
runs_seen: 5
switch_mode_at: 25000
switch_mode_to: grid
min_efficacy: 0.03
---

## Observation
re86: 42K-52K states explored across 5 runs. Zero levels solved. Ever.
Segment mode explores massively but learns nothing. Depth 150+.
The explorer has amnesia — re-explores the same dead ends every episode.

## Lesson
High state count + zero levels = exploring uniformly across dead ends.
Segment clicking finds states but none lead to solutions.
This game likely needs a fundamentally different interaction pattern (grid targeting,
specific coordinates, or action sequences rather than individual clicks).

## Action
Switch to grid mode by 25K actions if no levels. Try grid_fine after 50K.
Memory should help — if an action never changes the frame, stop doing it.
Budget 350K to give grid modes enough time.
