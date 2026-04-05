# Deepcopy Closure Bug

## The Problem
`copy.deepcopy()` on ARC-AGI-3 game objects breaks lambda closures. Games store opcode/action handlers as lambdas:
```python
self.dfguzecnsr = {3: lambda: self.otrzjnmayi(0, 4), ...}
```
After deepcopy, `self` in the lambda still references the ORIGINAL object, not the copy.

## Affected Games
- **TN36**: opcodes execute on wrong block (verified)
- **G50T**: 0 pixel change on any action after deepcopy
- **SC25**: 0 pixel change on any action, even with fresh instances

## Working Games
TU93, CD82, TR87, LS20, SB26, R11L, AR25, SP80, DC22, M0R0, KA59, SK48, FT09, LP85

## Workarounds
1. **Reset + replay**: create fresh game, replay action sequence for each BFS state (TN36 solver)
2. **Abstract solver**: bypass game engine entirely, BFS on pure tuples (LP85, TN36, FT09)
3. **Per-state dynamic actions**: use `_get_valid_clickable_actions()` on the ORIGINAL game (not copy)

## How to Test
```python
g = copy.deepcopy(game)
f0 = g.camera.render(g.current_level.get_sprites())
step_game(g, actions[0])
f1 = g.camera.render(g.current_level.get_sprites())
changed = (f0 != f1).sum()  # 0 = BROKEN
```
